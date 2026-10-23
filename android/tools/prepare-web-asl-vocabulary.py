"""Convert licensed 3D-LEX ASL motion onto AppHear's existing human skeleton.

Offline asset conversion, not training or linguistic validation. Old geometry,
binary animation data and all 258 actions are preserved. No source avatar mesh
or texture is imported. The new catalog is an explicit unreviewed preview.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import time
import zipfile

import numpy as np

SPEC = importlib.util.spec_from_file_location('web_hands', Path(__file__).with_name('audit-text-to-sign-hands.py'))
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)
R = H.r
ANDROID = Path(__file__).resolve().parents[1]
BASE_SHA = 'e9079308b6780e180528e4b893e64bdb77e46b33b6ad2cd99355bec03df57b18'
ARCHIVE_SHA = 'bac3e8ca0dbcc45eec0952a20dd04f28cc38131baddc8a354a72da673af0bdc7'
ASSET = 'asl_animations/web_vocabulary_200/avatar.glb'
REFERENCE = 'https://osf.io/mqzyk/'
# Alternative English inputs are deliberately not counted as new signs.
ALIASES = {
    'phone': ['telephone'], 'car': ['automobile'], 'buy': ['purchase'],
    'help': ['assist', 'assistance'], 'sick': ['ill', 'illness', 'unwell'],
    'teacher': ['instructor'], 'student': ['pupil'], 'neighbor': ['neighbour'],
    'work': ['labor', 'labour', 'employment'], 'road': ['street', 'path'],
    'mother': ['mommy', 'mummy'], 'father': ['daddy'], 'congratulations': ['congrats'],
}
PREFERRED = '''apple orange banana bread milk cheese egg breakfast lunch dinner food coffee tea juice soda soup sandwich salad pizza potato tomato carrot onion rice meat chicken fish beef pork fruit vegetable sugar salt pepper butter chocolate cookie candy ice cream cake kitchen bathroom bedroom living room garden beach mountain river lake ocean forest tree flower grass leaf sun moon star sky cloud wind snow weather summer winter spring fall monday tuesday wednesday thursday friday saturday sunday weekend night morning afternoon evening month year birthday party gift holiday vacation trip bus train airplane bicycle boat truck motorcycle taxi traffic airport station hotel library university college classroom office bank post office church park playground zoo museum supermarket nurse doctor dentist medicine pain hurt accident emergency police fire ambulance hospital health healthy exercise sport basketball football soccer baseball tennis swim dance sing music guitar piano violin art paint draw color red blue green yellow purple pink brown black white gray gold silver orange animal dog cat bird horse cow pig sheep goat duck rabbit mouse elephant lion tiger bear monkey snake frog turtle butterfly bee spider insect pet farm brother sister son daughter aunt uncle cousin niece nephew grandchild husband wife marriage wedding divorce baby child children adult young age birthday age family happy sad angry afraid scared surprised excited bored lonely confused proud nervous shy calm worry laugh smile cry fun funny serious kind polite rude patient smart stupid clever honest lie truth remember forget believe hope dream think know understand question answer explain describe story sentence language english sign deaf hearing interpreter translate practice study test homework lesson paper pencil pen notebook eraser scissors ruler desk chair table window door wall floor ceiling roof stair elevator key lock light lamp television radio camera picture photograph internet email text message computer phone screen battery charge broken fix repair build cut cook bake boil fry wash clean dirty wet dry hot warm cold cool hungry thirsty full empty tall short long thin fat heavy light hard soft loud quiet fast slow early late before after during next last first second third fourth five six seven eight nine ten hundred thousand million number count money dollar cent price cheap expensive free rich poor win lose game team together alone different same all none everyone nobody another other enough extra more less most few many much both each every always never sometimes often usually maybe possible impossible can cannot want need like dislike love hate prefer choose decide finish start continue stop wait ready busy tired sleep wake sit stand walk run jump climb throw catch push pull lift carry put take bring send receive borrow lend keep return arrive leave enter exit visit travel find search hide show teach learn read write listen hear see look watch taste smell touch feel work job earn pay buy sell help support give share save spend live die born grow change become happen make do act actor accomplish address adult agree against amaze announce appear area argue arm army audience avoid award awkward back bacon balance ball balloon bar barely basement bath beg behavior behind benefit bet bicycle birth blame blind blood bone borrow bottle bowl brain branch break bridge bright broken brush burn button cabbage calendar call cancel captain careful carpet cat catch ceiling center challenge chance check cheer chest child chin christmas circle city class clock close coat coconut collect color comb comfortable compare complain complete concern connect consider cook correct couple course court cover crash crazy credit cross crowd culture cup dangerous dark date december deep delicious dentist depend desert develop dictionary difficult direction dirty discover discuss disease divide doctor dollar dream dress drive dry duty ear earth east education eight empty engineer enough equal even event example experience eye face fail familiar far February fever field fight finger flag flat float fly focus follow food football foreign forever form four fresh friendly full future general gentle geography glove government grade gray grocery ground group guess hair half Halloween handsome happen harvest hate headache heart heavy height history honest human hundred hurry ice idea imagine improve include information injure inside interesting invite island January jealous July June keep kiss knife knee ladder late lazy left lemon length lesson letter level lie lift lip list little local loud low luck magic manage map March May meat memory metal middle minute miss model modern Monday monkey moon mouth movie multiply narrow nature necessary neck newspaper next noise normal north nose November October offer onion opinion opposite order other outside own page pair parent patient pay perfect perhaps pie pink place plan plant plenty pocket point popular possible postpone power precious president pretty price pride print process promise proof pull punish purple push put queen quiet rabbit race rake really red refuse relationship relax remember research responsibility rest result rich rock rude ruin salad sandwich Saturday scared schedule science scientist scold scream screwdriver secret secretary sensitive sentence sequence serious sew shame shape share shave shelf shirt shovel shower shrimp silly since sing situation ski skill skin smart smell smile smooth snake snow soap socks soda sofa soft sound south spoon spring square squirrel staff stairs stamp stare start steal sting stir straight strange stress struggle stubborn stuck study subtract summer sunrise surgery surprise suspect suspend swallow sweep sweet swing tan tea technology teeth tennis test text that therapy thermometer thin thirsty three Thursday ticket tiger tissue toast tobacco toilet tomato top tornado touch tournament towel train transfer tree trouble truth turtle two ugly uncle under unique university up vacation vampire verb violin visualize voice volunteer vomit vote waiter wall wander warn way weak weather wedding Wednesday weight wet whatever white will win wine with wolf wonder wood worry worse wow yellow young zero'''.split()

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def normalize(value: str) -> str:
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', value.lower().replace("'", '')).split())

def original_inventory() -> tuple[list[dict], dict, dict]:
    base = json.loads((ANDROID/'app/src/main/assets/asl_animations/manifest.json').read_text())
    batch = json.loads((ANDROID/'app/src/aslOriginalWords/assets/asl_animations/original_words_v1/manifest.json').read_text())
    registry = json.loads((ANDROID/'app/src/aslOriginalWords/assets/asl_animations/original_words_v1/fingerspelling-avatar.json').read_text())
    lookup = {e['id']: e for e in base['clips']}
    rows = []
    for entry in batch['replacements']+batch['newClips']:
        source = dict(lookup.get(entry['id'], {})); source.update(entry)
        source['english'] = source['aliases']
        rows.append(source)
    for entry in registry['wordAdditions']:
        rows.append(dict(entry, displayText=entry['english'][0]))
    alias = json.loads((ANDROID/'app/src/main/assets/asl_animations/existing_aliases_asl.json').read_text())
    by_id = {e['id']: e for e in rows}
    for entry in alias['aliases']+alias.get('webVocabularyAliases',[]):
        by_id[entry['signId']]['english'] = list(dict.fromkeys(by_id[entry['signId']]['english']+entry['english']))
    return rows, registry, batch

class Scene:
    def __init__(self, payload: bytes):
        assert struct.unpack_from('<II', payload) == (0x46546c67, 2)
        length = struct.unpack_from('<I', payload, 12)[0]
        self.g = json.loads(payload[20:20+length])
        self.binary = payload[28+length:]
        self.nodes = self.g['nodes']
        self.parents, self.order = R.skeleton(self.nodes)
        self.names = {n['name']: i for i,n in enumerate(self.nodes)}
        assert len(self.g['animations']) == 1
        self.tracks = []
        action = self.g['animations'][0]
        for ch in action['channels']:
            sample = action['samplers'][ch['sampler']]
            assert sample.get('interpolation', 'LINEAR') in ('LINEAR', 'STEP')
            times = self.data(sample['input']).flatten()
            values = self.data(sample['output'])
            assert np.isfinite(times).all() and np.isfinite(values).all() and np.all(np.diff(times)>0)
            if ch['target']['path']=='scale':
                # The publisher exports a uniform animated Hips calibration
                # scale (e.g. 1.04). It does not change anatomical directions.
                # Recover directions using it, but never transfer root scaling
                # or bone stretching to the existing AppHear character.
                name=self.nodes[ch['target']['node']]['name']
                assert name=='Hips' and np.max(np.ptp(values,axis=1))<2e-5 and np.min(values)>.5 and np.max(values)<2, 'Unsupported nonuniform/bone source scaling'
            self.tracks.append(dict(node=ch['target']['node'], path=ch['target']['path'], times=times.tolist(), values=values.tolist(), interpolation=sample.get('interpolation','LINEAR')))
        self.full_duration = max(t['times'][-1] for t in self.tracks)
        assert .35 < self.full_duration <= 12, 'Unsupported source duration'
        # Some publisher GLBs append a reset whose tip offsets jump from ~2 cm
        # to ~70 cm. Detect the mechanically invalid suffix, never guess a sign
        # boundary or alter an interior frame. Preserve exact source/time evidence.
        timeline = max(self.tracks, key=lambda t: len(t['times']))['times']
        invalid = np.zeros(len(timeline), dtype=bool)
        self.calibration_nodes = copy.deepcopy(self.nodes)
        for track in self.tracks:
            if track['path'] != 'translation' or self.nodes[track['node']]['name'] == 'Hips': continue
            assert track['times'] == timeline, 'Inconsistent source timelines'
            values = np.asarray(track['values'])
            typical = np.median(values, axis=0)
            self.calibration_nodes[track['node']]['translation'] = typical.tolist()
            length = np.linalg.norm(typical)
            if length > .001:
                ratio = np.linalg.norm(values, axis=1) / length
                invalid |= (ratio < .75) | (ratio > 1.25)
        bad = np.flatnonzero(invalid)
        self.excluded_reset = None
        self.duration = self.full_duration
        if len(bad):
            first = int(bad[0])
            assert first > 2 and len(bad) <= max(4, int(len(timeline)*.05)), 'Invalid source bone lengths outside a short export suffix'
            assert np.all(invalid[first:]), 'Invalid interior source bone lengths'
            self.duration = timeline[first-1]
            self.excluded_reset = dict(reason='Publisher reset suffix: non-root bone length outside 75–125% of its median',
                firstExcludedSeconds=timeline[first], lastExcludedSeconds=timeline[-1], excludedFrames=len(bad),
                retainedThroughSeconds=self.duration, sourceBytesUnchanged=True)

    def data(self, index: int) -> np.ndarray:
        a=self.g['accessors'][index]; v=self.g['bufferViews'][a['bufferView']]
        assert 'sparse' not in a
        dtype={5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']]
        n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        return np.ndarray((a['count'],n),dtype=dtype,buffer=self.binary,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',np.dtype(dtype).itemsize*n),np.dtype(dtype).itemsize)).copy()

    def pose(self, time_sec: float) -> dict:
        return R.world(self.nodes,self.parents,self.order,{(t['node'],t['path']):R.sample(t,time_sec) for t in self.tracks})

def body_frame(world: dict, names: dict) -> np.ndarray:
    up=world[names['Spine2']][:3,3]-world[names['Hips']][:3,3]
    across=world[names['LeftArm']][:3,3]-world[names['RightArm']][:3,3]
    return R.frame(up,np.cross(across,up))

def palm(world: dict, names: dict, side: str) -> np.ndarray:
    return np.cross(world[names[side+'HandIndex1']][:3,3]-world[names[side+'HandPinky1']][:3,3],world[names[side+'HandMiddle1']][:3,3]-world[names[side+'Hand']][:3,3])

def pairs() -> dict[str,str]:
    result={'Hips':'Spine','Spine':'Spine1','Spine1':'Spine2','Spine2':'Neck','Neck':'Head','Head':'HeadTop_End'}
    for side in ('Left','Right'):
        for a,b in [('Shoulder','Arm'),('Arm','ForeArm'),('ForeArm','Hand'),('Hand','HandMiddle1'),('UpLeg','Leg'),('Leg','Foot'),('Foot','ToeBase'),('ToeBase','Toe_End')]: result[side+a]=side+b
        for f in ('Thumb','Index','Middle','Ring','Pinky'):
            for i in range(1,4): result[side+'Hand'+f+str(i)]=side+'Hand'+f+str(i+1)
    return result

class Converter:
    def __init__(self, model):
        self.m=model
        self.neutral={ (t['node'],t['path']):R.sample(t,0) for t in model.tracks('SIGN_RELAXED_NEUTRAL') if t['path']=='rotation'}
        self.rest=R.world(model.nodes,model.parents,model.order,self.neutral)
        self.pairs=pairs()
        self.target_frame={name:self.frame(self.rest,model.names,name,child) for name,child in self.pairs.items()}
        self.neutral_body=body_frame(self.rest,model.names)

    @staticmethod
    def frame(world: dict,names: dict,name: str,child: str) -> np.ndarray:
        direction=world[names[child]][:3,3]-world[names[name]][:3,3]
        if 'Hand' in name or name.endswith(('Arm','ForeArm')):
            normal=palm(world,names,'Left' if name.startswith('Left') else 'Right')
        elif name=='Head':
            normal=np.cross(world[names['LeftEye']][:3,3]-world[names['RightEye']][:3,3],direction)
        else: normal=body_frame(world,names)[:,2]
        if np.linalg.norm(np.cross(direction,normal))<1e-7:
            normal=R.orient(world[names[name]])[:,2]
        assert np.linalg.norm(np.cross(direction,normal))>1e-8
        return R.frame(direction,normal)

    def convert(self, scene: Scene) -> tuple[np.ndarray,list[dict],dict]:
        m=self.m
        assert set(self.pairs).issubset(scene.names) and set(self.pairs.values()).issubset(scene.names)
        initial=scene.pose(0)
        alignment=self.neutral_body@body_frame(initial,scene.names).T
        bind=R.world(scene.calibration_nodes,scene.parents,scene.order)
        # Calibrate bone-local axes once. Using a moving palm normal to solve
        # forearm/finger twist every frame creates artificial 180-degree flips.
        correction={name:R.orient(bind[scene.names[name]]).T@
            self.frame(bind,scene.names,name,child)@self.target_frame[name].T@R.orient(self.rest[m.names[name]])
            for name,child in self.pairs.items()}
        times=np.linspace(0,scene.duration,max(3,int(np.ceil(scene.duration*24))+1))
        frames=[]; wrists=[]; max_error=0.
        for time_sec in times:
            sw=scene.pose(float(time_sec)); desired={}
            for name,child in self.pairs.items():
                t=m.names[name]
                desired[t]=alignment@R.orient(sw[scene.names[name]])@correction[name]
            current={}; out={}
            for i in m.order:
                node=m.nodes[i]; parent=current.get(m.parents.get(i),np.eye(4))
                q=self.neutral.get((i,'rotation'),node.get('rotation',[0,0,0,1]))
                if i in desired: q=R.quaternion(R.orient(parent).T@desired[i])
                out[(i,'rotation')]=np.asarray(q)
                local=np.eye(4);local[:3,:3]=R.rotation(q)@np.diag(node.get('scale',[1,1,1]));local[:3,3]=node.get('translation',[0,0,0])
                if 'matrix' in node:local=np.array(node['matrix']).reshape(4,4,order='F')
                current[i]=parent@local
            assert all(np.isfinite(q).all() for q in out.values())
            # Independent directional constraint: no generic pose is substituted.
            for name,child in self.pairs.items():
                wanted=R.unit(alignment@(sw[scene.names[child]][:3,3]-sw[scene.names[name]][:3,3]))
                got=R.unit(current[m.names[child]][:3,3]-current[m.names[name]][:3,3])
                max_error=max(max_error,float(np.linalg.norm(wanted-got)))
            wrists.append(np.r_[current[m.names['LeftHand']][:3,3],current[m.names['RightHand']][:3,3]])
            frames.append(out)
        # Source translations include sub-millimetre mocap/export noise; retain
        # the orientation tracks with fixed target bone lengths, within 1 degree.
        assert max_error<.01745, ('Bone direction transfer exceeds one degree',max_error)
        span=float(np.max(np.ptp(wrists,axis=0)))
        finger_variation=max(float(np.max(np.linalg.norm(np.asarray([f[(i,'rotation')] for f in frames])-frames[0][(i,'rotation')],axis=1))) for name,i in m.names.items() if 'Hand' in name and not name.endswith('4'))
        assert span>.015 or finger_variation>.12, 'Static/placeholder source motion'
        inner=np.asarray([[f[(i,'rotation')] for i in m.order] for f in frames])
        steps=2*np.arccos(np.clip(np.abs(np.sum(inner[:-1]*inner[1:],axis=2)),0,1))
        max_step=float(np.max(steps))
        assert max_step<np.pi/2, ('Abrupt source/retarget rotation exceeds 90 degrees per 24 Hz frame',float(np.degrees(max_step)))
        # New clips explicitly enter/leave the old neutral, so prior sign state
        # cannot leak into them. Source duration/motion itself is not shortened.
        neutral={ (i,'rotation'):np.asarray(self.neutral.get((i,'rotation'),m.nodes[i].get('rotation',[0,0,0,1]))) for i in m.order }
        output_times=np.r_[0.,times+.2,scene.duration+.4].astype('<f4')
        return output_times,[neutral]+frames+[neutral],{'source_full_duration_seconds':scene.full_duration,'retained_source_duration_seconds':scene.duration,'excluded_export_reset':scene.excluded_reset,'sampled_source_frames':len(times),'max_direction_error':max_error,'max_inner_rotation_step_degrees':float(np.degrees(max_step)),'wrist_span_m':span,'finger_rotation_variation':finger_variation,'source_translations':'Median non-root offsets calibrate bone-local axes; animated orientation tracks transferred, target bone lengths retained','facial_blendshapes':'Not present in this GLB motion release; not invented'}

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--count',type=int,default=200)
    args=parser.parse_args();start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'avatar.glb').exists(), 'Use a fresh output directory'
    archive_data=args.archive.read_bytes();assert digest(archive_data)==ARCHIVE_SHA
    baseline=ANDROID/'app/src/aslOriginalWords/assets/asl_animations/original_words_v1/fingerspelling-avatar.glb'
    payload=baseline.read_bytes();assert digest(payload)==BASE_SHA
    old,_,_=original_inventory();assert len(old)==232
    occupied={normalize(s) for entry in old for s in entry['english']}
    occupied.update(normalize(s) for forms in ALIASES.values() for s in forms)
    occupied.update({'we','they','fine','large','favourite','granddad','grandad','granny','alone','job','woman','man'})
    model=H.Model(baseline);converter=Converter(model)
    g=copy.deepcopy(model.g);binary=bytearray(model.binary[:g['buffers'][0]['byteLength']]);preserved=len(g['animations']);assert preserved==258
    def accessor(values: np.ndarray,kind: str) -> int:
        values=np.asarray(values,dtype='<f4');assert np.isfinite(values).all()
        while len(binary)%4:binary.append(0)
        view=len(g['bufferViews']);g['bufferViews'].append(dict(buffer=0,byteOffset=len(binary),byteLength=values.nbytes));binary.extend(values.tobytes())
        index=len(g['accessors']);item=dict(bufferView=view,componentType=5126,count=len(values),type=kind)
        if kind=='SCALAR':item.update(min=[float(values.min())],max=[float(values.max())])
        g['accessors'].append(item);return index
    accepted=[];rejected=[]
    with zipfile.ZipFile(args.archive) as archive:
        members=[e for e in archive.infolist() if e.filename.endswith('.glb')]
        duplicate=collections.Counter((e.CRC,e.file_size) for e in members)
        eligible={Path(e.filename).stem:e for e in members if re.fullmatch('[a-z]+',Path(e.filename).stem) and normalize(Path(e.filename).stem) not in occupied and duplicate[(e.CRC,e.file_size)]==1}
        planned=list(dict.fromkeys([s for s in PREFERRED if s in eligible]+sorted(eligible)))
        source_hashes=set();motion_hashes=set()
        for word in planned:
            entry=eligible[word];source_payload=archive.read(entry)
            source_sha=digest(source_payload)
            if source_sha in source_hashes: continue
            try:
                source=Scene(source_payload);times,frames,checks=converter.convert(source)
                # Digest the converted motion rather than trusting different filenames.
                motion_bytes=np.asarray([[f[(i,'rotation')] for i in model.order] for f in frames],dtype='<f4').tobytes()
                motion_sha=digest(motion_bytes)
                assert motion_sha not in motion_hashes, 'Duplicate converted motion'
            except (AssertionError,KeyError,ValueError,FloatingPointError) as error:
                rejected.append({'word':word,'source':entry.filename,'reason':str(error)})
                continue
            name=f'SIGN_{word.upper()}_3DLEX_ASL_PREVIEW'
            action=dict(name=name,samplers=[],channels=[]);inp=accessor(times,'SCALAR')
            for i in model.order:
                if 'matrix' in model.nodes[i]:continue
                values=np.asarray([f[(i,'rotation')] for f in frames])
                for j in range(1,len(values)):
                    if values[j-1]@values[j]<0:values[j]*=-1
                si=len(action['samplers']);action['samplers'].append(dict(input=inp,output=accessor(values,'VEC4'),interpolation='LINEAR'));action['channels'].append(dict(sampler=si,target=dict(node=i,path='rotation')))
            g['animations'].append(action)
            accepted.append(dict(id=word,english=[word],animationName=name,durationMs=int(round(float(times[-1])*1000)),
                sourceArchiveMember=entry.filename,sourceSha256=source_sha,motionSha256=motion_sha,
                sourceLanguage='ASL',sourceProject='3D-LEX v1.0 ASL Animation Data',sourceUrl=REFERENCE,
                license='CC-BY-4.0',construction='ANATOMICAL_FRAME_RETARGETED_PREVIEW',
                linguisticValidationStatus='UNREVIEWED',technicalPlaybackStatus='Offline transform checks only; device review pending',checks=checks))
            source_hashes.add(source_sha);motion_hashes.add(motion_sha)
            if len(accepted)%10==0 or len(accepted)==args.count:print(json.dumps({'accepted':len(accepted),'rejected':len(rejected),'seconds':round(time.monotonic()-start,2)}),flush=True)
            if len(accepted)==args.count:break
        (args.output/'candidate_rejections.json').write_text(json.dumps(rejected,indent=2)+'\n')
        assert len(accepted)==args.count,(len(accepted),args.count,rejected[-5:])
    for key in ['nodes','skins','meshes','materials','textures','images','scenes']:assert g[key]==model.g[key],key
    assert g['animations'][:preserved]==model.g['animations']
    assert binary[:model.g['buffers'][0]['byteLength']]==model.binary[:model.g['buffers'][0]['byteLength']]
    g['buffers'][0]['byteLength']=len(binary);encoded=json.dumps(g,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%4);binary+=b'\0'*(-len(binary)%4)
    output=struct.pack('<III',0x46546c67,2,28+len(encoded)+len(binary))+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary
    (args.output/'avatar.glb').write_bytes(output)
    manifest=dict(schemaVersion=1,packageId='3dlex-asl-preview-200-v1',language='ASL',avatarProfile='ORIGINAL_HUMAN',
        asset=ASSET,sha256=digest(output),baselineSha256=BASE_SHA,originalActions=preserved,newLexicalEntries=len(accepted),
        totalLexicalEntries=232+len(accepted),status='UNREVIEWED_RETARGETED_PREVIEW',qualifiedAslValidations=0,
        license='CC-BY-4.0 for 3D-LEX motion; existing avatar rights remain separately unresolved',
        attribution='Oline Ranum, Gomer Otterspeer, Jari I. Andersen, Robert G. Belleman and Floris Roelofsen (2024). 3D-LEX v1.0. ASL subset. https://osf.io/cv276/ . Adapted to AppHear skeleton; source meshes/textures not copied.',
        sourceArchiveSha256=ARCHIVE_SHA,entries=accepted)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(status='OFFLINE_CONVERSION_COMPLETE_NOT_LINGUISTIC_VALIDATION',baselineSha256=BASE_SHA,
        archiveSha256=ARCHIVE_SHA,preservedActions=preserved,newEntries=len(accepted),totalActions=len(g['animations']),
        unchangedGeometryAndOriginalBinary=True,qualifiedAslValidations=0,sourceDuplicatePayloadsExcluded=True,
        rejectedCandidates=rejected,seconds=round(time.monotonic()-start,3),bundleBytes=len(output),bundleSha256=digest(output))
    (args.output/'conversion_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
