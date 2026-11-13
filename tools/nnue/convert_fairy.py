import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import onnx
import onnxruntime as ort

parser=argparse.ArgumentParser(description="Convert the verified Fairy xiangqi network to ONNX; native reference is offline only.")
parser.add_argument("--input",type=Path,required=True)
parser.add_argument("--reference-engine",type=Path,required=True)
parser.add_argument("--samples",type=Path,required=True)
parser.add_argument("--output-dir",type=Path,required=True)
parser.add_argument("--positions",type=int,default=64)
args=parser.parse_args()
HOME=args.output_dir.resolve()
HOME.mkdir(parents=True,exist_ok=True)
if any(HOME.iterdir()):raise FileExistsError("Use an empty output directory to preserve experiment evidence")
(HOME/"converter.snapshot.py").write_bytes(Path(__file__).read_bytes())
torch.set_num_threads(4)
raw=args.input.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='c07e94a5c7cbeae443ed79a8fa412875d833a7f8e04333815e39729c59d52e11'
offset=0
def take(dtype,count):
    global offset
    result=np.frombuffer(raw,dtype=dtype,count=count,offset=offset).copy();offset+=result.nbytes;return result
def u32():return int(take('<u4',1)[0])
version,network_hash,description_len=u32(),u32(),u32()
description=raw[offset:offset+description_len].decode();offset+=description_len
assert version==0x7af32f20
feature_hash=u32();assert feature_hash==0x5f2348b8
ft_bias=take('<i2',512).astype(np.float32)
ft_weight=take('<i2',10530*512).reshape(10530,512).astype(np.float32)
psqt=take('<i4',10530*8).reshape(10530,8).astype(np.float32)
layers=[[],[],[]]
for bucket in range(8):
    assert u32()==network_hash^feature_hash
    for index,(inputs,outputs,padded) in enumerate([(1024,16,1024),(16,32,32),(32,1,32)]):
        bias=take('<i4',outputs).astype(np.float32)
        weight=take('i1',outputs*padded).reshape(outputs,padded).astype(np.float32)
        assert not np.any(weight[:,inputs:])
        layers[index].append((weight[:,:inputs],bias))
assert offset==len(raw),(offset,len(raw))

def features(row):
    flat=np.asarray(row['BoardEncoding']).reshape(90,14)
    pieces=[(sq,plane) for sq in range(90) for plane in range(14) if flat[sq,plane]]
    kings=[next(sq for sq,plane in pieces if plane==side*7) for side in range(2)]
    # Core types: king/advisor/elephant/horse/rook/cannon/soldier.
    # Fairy piece-type enumeration: rook/advisor/cannon/soldier/horse/elephant/king.
    groups=[6,1,5,4,0,2,3]
    def perspective(side):
        krow,kcol=divmod(kings[side],9);krank=(9-krow) if side==0 else krow
        assert 0<=krank<=2 and 3<=kcol<=5
        base=(krank*3+kcol-3)*1170
        values=[]
        for sq,plane in pieces:
            r,c=divmod(sq,9);rank=(9-r) if side==0 else r
            kind=plane%7
            group=12 if kind==0 else groups[kind]*2+int(plane//7!=side)
            values.append(base+group*90+rank*9+c)
        result=np.full(32,10530,np.int64);result[:len(values)]=sorted(values);return result
    stm=0 if row['SideToMove']==1 else 1
    return perspective(stm),perspective(1-stm),min((len(pieces)-1)*8//32,7)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        for name,array in [('ft',np.concatenate((ft_weight,np.zeros((1,512),np.float32)))),('bias',ft_bias),('psqt',np.concatenate((psqt,np.zeros((1,8),np.float32))))]:
            self.register_buffer(name,torch.from_numpy(array))
        for i,layer in enumerate(layers):
            self.register_buffer(f'w{i}',torch.from_numpy(np.stack([p[0] for p in layer])))
            self.register_buffer(f'b{i}',torch.from_numpy(np.stack([p[1] for p in layer])))
    def forward(self,us,them,bucket):
        a=(F.embedding(us,self.ft).sum(1)+self.bias).clamp(0,127)
        b=(F.embedding(them,self.ft).sum(1)+self.bias).clamp(0,127)
        x=torch.cat((a,b),1)
        for i in range(3):
            x=torch.bmm(getattr(self,f'w{i}')[bucket],x.unsqueeze(2)).squeeze(2)+getattr(self,f'b{i}')[bucket]
            if i<2:x=torch.floor(x/64).clamp(0,127)
        p=(self.psqt[us].sum(1)-self.psqt[them].sum(1)).gather(1,bucket[:,None]).squeeze(1)
        half_p=torch.sign(p)*torch.floor(torch.abs(p)/2)
        summed=(half_p+x.squeeze(1))/16
        raw_value=torch.sign(summed)*torch.floor(torch.abs(summed))
        return raw_value*100/208

rows=[]
with args.samples.open(encoding='utf-8-sig') as stream:
    for line in stream:
        row=json.loads(line)
        if row['MoveIndex']%3==0:rows.append(row)
        if len(rows)>=args.positions:break
us,them,bucket=map(np.stack,zip(*(features(row) for row in rows)))
model=Net().eval()
with torch.no_grad(): prediction=model(torch.from_numpy(us),torch.from_numpy(them),torch.from_numpy(bucket)).numpy()

# Independent native reference is OFFLINE only. It never selects a candidate move.
proc=subprocess.Popen([str(args.reference_engine.resolve())],cwd=HOME,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
def send(s):proc.stdin.write(s+'\n');proc.stdin.flush()
def until(marker):
    lines=[]
    while True:
        line=proc.stdout.readline()
        if not line:raise RuntimeError('Reference exited')
        lines.append(line)
        if line.strip()==marker:return ''.join(lines)
send('uci');until('uciok');send('setoption name UCI_Variant value xiangqi');send('setoption name EvalFile value '+str(args.input.resolve()));send('isready');until('readyok')
comparisons=[]
with (HOME/'native-reference.log').open('w') as log:
    for i,row in enumerate(rows):
        send('position fen '+row['Fen']);send('eval');send('isready');trace=until('readyok');log.write(trace)
        match=re.search(r'NNUE evaluation\s+([+-]?[\d.]+)',trace)
        if not match:continue
        reference=float(match[1])*100*row['SideToMove']
        comparisons.append(dict(index=i,side=row['SideToMove'],reference_cp=reference,python_cp=float(prediction[i]),error=abs(reference-float(prediction[i]))))
send('quit');proc.wait(timeout=5)
(HOME/'native-parity.json').write_text(json.dumps(comparisons,indent=2))
assert len(comparisons)>=32
error=max(item['error'] for item in comparisons)
assert error<=0.51,('Native parity failed',error,comparisons[:4])
path=HOME/'fairy-value.onnx'
torch.onnx.export(model,(torch.from_numpy(us[:1]),torch.from_numpy(them[:1]),torch.from_numpy(bucket[:1])),str(path),
                  input_names=['us','them','bucket'],output_names=['value_cp'],opset_version=18,dynamo=False,
                  dynamic_axes={'us':{0:'batch'},'them':{0:'batch'},'bucket':{0:'batch'},'value_cp':{0:'batch'}})
graph=onnx.load(path);onnx.helper.set_model_props(graph,{'encoding':'xiangqi-halfka-v2-10530-v1','score':'current-player-centipawns','source_sha256':hashlib.sha256(raw).hexdigest()});onnx.save(graph,path);onnx.checker.check_model(graph)
session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
actual=session.run(None,dict(us=us,them=them,bucket=bucket))[0]
ort_error=float(np.max(np.abs(actual-prediction)));assert ort_error<0.01,ort_error
report=dict(description=description,parsed_bytes=offset,native_positions=len(comparisons),native_max_cp_error=error,onnx_max_cp_error=ort_error,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(HOME/'conversion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
