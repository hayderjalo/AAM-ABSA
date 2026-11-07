import os, json, random, hashlib, time, gc, shutil, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate AAM-ABSA configurations.")
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing absa_train.csv, absa_validation.csv, absa_test.csv')
    parser.add_argument('--image-dir', type=Path, required=True, help='Root directory containing input images')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--variants', nargs='+', choices=['AMID_HCF','no_AMID','no_HCF','text_only'], default=['AMID_HCF','no_AMID','no_HCF','text_only'])
    parser.add_argument('--label-provenance', default=None, help='Optional description supplied by the data owner')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    root = args.data_dir.resolve()
    OUT = args.output_dir.resolve(); OUT.mkdir(parents=True, exist_ok=True)
    DEVICE=args.device; SEED=args.seed; EPOCHS=args.epochs; MAXLEN=96; D=128
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    from validate_data import validate
    validate(root, args.image_dir)
    frames=[pd.read_csv(root/f'absa_{s}.csv') for s in ['train','validation','test']]
    print('SPLITS', [len(x) for x in frames], flush=True)
    def decode(v):
        a=json.loads(v) if isinstance(v,str) else []
        if isinstance(a,dict): a=a.get('aspects',a.get('labels',[]))
        return sorted({str(q.get('aspect',q.get('category')))+'::'+str(q.get('sentiment',q.get('polarity'))) for q in a if q.get('aspect',q.get('category')) and q.get('sentiment',q.get('polarity'))})
    labs=[[decode(v) for v in f.aspect_labels_json.fillna('[]')] for f in frames]
    train_aspects=sorted({q.rsplit('::',1)[0] for row in labs[0] for q in row})
    classes=[a+'::'+s for a in train_aspects for s in ['negative','neutral','positive']]
    mlb=MultiLabelBinarizer(classes=classes); mlb.fit(labs[0]); classes=list(mlb.classes_)
    assert classes, 'No joint labels parsed'
    unknown=sorted({q for ls in labs[1:] for row in ls for q in row}-set(classes))
    assert not unknown, f'Unseen validation/test labels: {unknown}'
    y=[torch.tensor(mlb.transform(ls),dtype=torch.float32) for ls in labs]
    image_root = args.image_dir.resolve()
    def image_path(row):
        name = str(row.image_filename)
        p = (image_root / name).resolve()
        if not p.is_relative_to(image_root) or not p.is_file():
            raise ValueError('image_filename must resolve to a file inside --image-dir')
        return p
    ipaths=[[image_path(r) for _,r in f.iterrows()] for f in frames]
    # Audit split overlap. Existing splits are preserved and limitations are reported.
    texts=[set(f.text.astype(str).str.strip()) for f in frames]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for ps in ipaths for p in ps}
    image_sets=[{hashes[str(p)] for p in ps} for ps in ipaths]
    audit={'rows':dict(zip(['train','validation','test'],map(len,frames))), 'joint_labels':classes,'reference_labels':args.label_provenance,'text_overlap_train_test':len(texts[0]&texts[2]),'image_hash_overlap_train_test':len(image_sets[0]&image_sets[2]),'image_hash_overlap_train_validation':len(image_sets[0]&image_sets[1])}
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    print('AUDIT',audit,flush=True)
    MODEL='aubmindlab/bert-base-arabertv02'
    tok=AutoTokenizer.from_pretrained(MODEL); bert=AutoModel.from_pretrained(MODEL).to(DEVICE).eval()
    encoded=[tok(f.text.astype(str).tolist(),max_length=MAXLEN,truncation=True,padding='max_length',return_tensors='pt') for f in frames]
    # Frozen pretrained feature extraction is explicit and identical across ablations.
    features=[]
    with torch.inference_mode():
        for split,e in enumerate(encoded):
            parts=[]
            for start in range(0,len(e['input_ids']),32):
                z=bert(input_ids=e['input_ids'][start:start+32].to(DEVICE),attention_mask=e['attention_mask'][start:start+32].to(DEVICE)).last_hidden_state
                parts.append(z.cpu().half())
                if start%320==0: print('BERT cache',split,start,flush=True)
            features.append(torch.cat(parts))
    embedding=bert.get_input_embeddings().weight.detach().cpu().clone()
    del bert; gc.collect(); torch.cuda.empty_cache()
    vision=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    vision=nn.Sequential(*list(vision.children())[:-2]).to(DEVICE).eval()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    visual_cache={}
    with torch.inference_mode():
        unique=list(dict.fromkeys(str(p) for ps in ipaths for p in ps))
        for start in range(0,len(unique),32):
            names=unique[start:start+32]
            ims=[]
            for name in names:
                with Image.open(name) as im: ims.append(tf(im.convert('RGB')))
            z=vision(torch.stack(ims).to(DEVICE)).flatten(2).transpose(1,2).cpu().half()
            visual_cache.update(zip(names,z))
            if start%320==0: print('Image cache',start,flush=True)
    del vision; gc.collect(); torch.cuda.empty_cache()
    visual=[torch.stack([visual_cache[str(p)] for p in ps]) for ps in ipaths]
    loaders=[DataLoader(TensorDataset(e['input_ids'],e['attention_mask'],f,v,target),batch_size=32,shuffle=i==0) for i,(e,f,v,target) in enumerate(zip(encoded,features,visual,y))]
    class AMID(nn.Module):
        def __init__(self):
            super().__init__(); self.detect=nn.Linear(D,D); self.filter=nn.Linear(D,D); self.refine=nn.Sequential(nn.Linear(D,D),nn.GELU(),nn.Linear(D,D)); self.alpha=nn.Parameter(torch.tensor(.1))
        def forward(self,F):
            M1=torch.sigmoid(self.detect(F)); M2=torch.sigmoid(self.filter(F*M1)); F2=F*M2
            return F2+self.alpha*self.refine(F2)
    class Network(nn.Module):
        def __init__(self,variant):
            super().__init__(); self.variant=variant
            self.embedding=nn.Embedding.from_pretrained(embedding,freeze=True)
            self.lstm=nn.LSTM(embedding.shape[1],D//2,batch_first=True,bidirectional=True)
            self.hybrid=nn.LSTM(features[0].shape[-1],D//2,batch_first=True,bidirectional=True)
            self.bp=nn.Linear(features[0].shape[-1],D); self.vp=nn.Linear(512,D)
            self.combine=nn.Linear(3*D,D); self.at=AMID(); self.av=AMID()
            self.st=nn.MultiheadAttention(D,4,batch_first=True); self.sv=nn.MultiheadAttention(D,4,batch_first=True)
            self.tv=nn.MultiheadAttention(D,4,batch_first=True); self.vt=nn.MultiheadAttention(D,4,batch_first=True)
            self.nt=nn.LayerNorm(D); self.nv=nn.LayerNorm(D)
            self.ffn=nn.Sequential(nn.Linear(4*D,D),nn.GELU(),nn.Dropout(.2)); self.simple=nn.Linear(2*D,D); self.head=nn.Linear(D,len(classes))
        def forward(self,ids,mask,b,v):
            lens=mask.sum(1).cpu(); emb=self.embedding(ids)
            packed=nn.utils.rnn.pack_padded_sequence(emb,lens,batch_first=True,enforce_sorted=False)
            a,_=self.lstm(packed); a,_=nn.utils.rnn.pad_packed_sequence(a,batch_first=True,total_length=ids.shape[1])
            packed=nn.utils.rnn.pack_padded_sequence(b.float(),lens,batch_first=True,enforce_sorted=False)
            h,_=self.hybrid(packed); h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=ids.shape[1])
            t=self.combine(torch.cat([a,self.bp(b.float()),h],-1)); v=self.vp(v.float()); pad=~mask.bool()
            if self.variant!='no_AMID': t=self.at(t); v=self.av(v)
            def pool(x): return (x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            if self.variant=='text_only': return self.head(pool(t))
            if self.variant=='no_HCF': return self.head(self.simple(torch.cat([pool(t),v.mean(1)],-1)))
            st,_=self.st(t,t,t,key_padding_mask=pad,need_weights=False); sv,_=self.sv(v,v,v,need_weights=False)
            t=self.nt(t+st); v=self.nv(v+sv)
            ct,_=self.tv(t,v,v,need_weights=False); cv,_=self.vt(v,t,t,key_padding_mask=pad,need_weights=False)
            return self.head(self.ffn(torch.cat([pool(ct),cv.mean(1),pool(t),v.mean(1)],-1)))
    def scores(g,p):
        pr,re,f,_=precision_recall_fscore_support(g,p,average='macro',zero_division=0)
        mi=precision_recall_fscore_support(g,p,average='micro',zero_division=0)[2]
        return {'Joint Macro-F1':float(f),'Joint Micro-F1':float(mi),'Macro-Precision':float(pr),'Macro-Recall':float(re),'Joint Accuracy':float(accuracy_score(g,p))}
    def infer(model,loader):
        model.eval(); probs=[]; gold=[]
        with torch.no_grad():
            for ids,m,b,v,target in loader:
                logits=model(ids.to(DEVICE),m.to(DEVICE),b.to(DEVICE),v.to(DEVICE)); probs.append(logits.sigmoid().cpu().numpy()); gold.append(target.numpy())
        return np.concatenate(probs),np.concatenate(gold).astype(int)
    config={'seed':SEED,'epochs':EPOCHS,'batch':32,'max_length':MAXLEN,'pretrained_text':MODEL,'pretrained_image':'ResNet18 ImageNet','pretrained_encoders':'frozen cached features; trainable BiLSTM, hybrid BiLSTM, AMID and HCF','AMID_M2':'sigmoid(W2*(F*M1)+b2)','joint_accuracy':'exact match across all aspect-sentiment pairs','splits':'supplied splits; see data_audit.json for overlap','label_reference':audit['reference_labels'],'status':'running'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    results=[]
    for variant in args.variants:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        model=Network(variant).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=3e-4); lossfn=nn.BCEWithLogitsLoss(); best=-1; history=[]
        folder=OUT/variant; folder.mkdir(exist_ok=True)
        for epoch in range(EPOCHS):
            model.train(); total=0.; n=0; started=time.time()
            for step,(ids,m,b,v,target) in enumerate(loaders[0]):
                opt.zero_grad(set_to_none=True); logits=model(ids.to(DEVICE),m.to(DEVICE),b.to(DEVICE),v.to(DEVICE)); loss=lossfn(logits,target.to(DEVICE)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=loss.item()*len(ids); n+=len(ids)
                if step%100==0: print(variant,'epoch',epoch+1,'batch',step,'loss',loss.item(),flush=True)
            pv,g= infer(model,loaders[1]); val,threshold=max((scores(g,(pv>=t).astype(int))['Joint Macro-F1'],float(t)) for t in np.arange(.1,.81,.05))
            row={'epoch':epoch+1,'train_loss':total/n,'validation_macro_f1':val,'threshold':threshold,'seconds':time.time()-started}; history.append(row)
            pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False); print(variant,row,flush=True)
            if val>best:
                best=val; torch.save({'state_dict':model.state_dict(),'threshold':threshold,'classes':classes,'variant':variant,'config':config},folder/'best.pt')
        ck=torch.load(folder/'best.pt',map_location=DEVICE,weights_only=False); model.load_state_dict(ck['state_dict']); p,g=infer(model,loaders[2]); pred=(p>=ck['threshold']).astype(int)
        result={'variant':variant,**scores(g,pred),'threshold':ck['threshold'],'test_rows':len(g),'best_validation_macro_f1':best}; results.append(result)
        (folder/'metrics.json').write_text(json.dumps(result,indent=2)); np.savez_compressed(folder/'predictions.npz',probabilities=p,gold=g,predictions=pred,classes=np.array(classes))
        per=precision_recall_fscore_support(g,pred,average=None,zero_division=0)
        pd.DataFrame({'label':classes,'precision':per[0],'recall':per[1],'f1':per[2],'support':per[3]}).to_csv(folder/'per_label_metrics.csv',index=False)
        pd.DataFrame({'sample_id':frames[2].sample_id,'gold':['|'.join(np.array(classes)[row.astype(bool)]) for row in g],'predicted':['|'.join(np.array(classes)[row.astype(bool)]) for row in pred],'exact_match':(g==pred).all(1)}).to_csv(folder/'test_predictions.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history]); axes[0].set(title='Training loss',xlabel='Epoch'); axes[1].plot([r['epoch'] for r in history],[r['validation_macro_f1'] for r in history]); axes[1].set(title='Validation Joint Macro-F1',xlabel='Epoch'); fig.tight_layout(); fig.savefig(folder/'learning_curves.png',dpi=200); plt.close(fig)
        pd.DataFrame(results).to_csv(OUT/'ablation_results.csv',index=False); print('TEST',json.dumps(result),flush=True)
        del model,opt,ck; gc.collect(); torch.cuda.empty_cache()
    table=pd.DataFrame(results); fig,ax=plt.subplots(figsize=(9,5)); table.set_index('variant')[['Joint Macro-F1','Joint Micro-F1','Joint Accuracy']].plot.bar(ax=ax); ax.set_ylim(0,1); ax.set_ylabel('Score'); fig.tight_layout(); fig.savefig(OUT/'ablation_comparison.png',dpi=220); plt.close(fig)
    config['status']='completed'; (OUT/'config.json').write_text(json.dumps(config,indent=2)); (OUT/'FINAL_RESULTS.json').write_text(json.dumps(results,indent=2))
    shutil.make_archive(str(OUT.parent / (OUT.name + '_results')), 'zip', OUT)
    print('ALL TRAINING AND TEST EVALUATION COMPLETED',flush=True)
    print(table.to_string(index=False),flush=True)

if __name__ == "__main__":
    main()
