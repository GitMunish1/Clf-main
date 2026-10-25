"""Isolated synthetic software check; never starts FastAPI or trains a model.

Usage: python scripts/verify_backend_contract.py /path/to/clf-backend
The temporary status=trained declaration below is TEST METADATA used solely to
exercise the real loader. The network weights remain untrained throughout.
"""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from model.clf_artifact import build_clf_model, make_feature_contract, make_grid_metadata, write_untrained_package
from app.models.model_runtime import ModelRuntime
from app.services.preprocessing import FeaturePipeline
from app.schemas.sensors import SensorPacket
from uuid import uuid4

with tempfile.TemporaryDirectory(prefix='clf-mcel-serialization-test-') as directory:
    contract=make_feature_contract([f'02:00:00:00:00:{i:02x}' for i in range(24)])
    grid=make_grid_metadata([{'floor':f,'width_m':3,'height_m':3} for f in (0,1)],3,.3)
    calibration={'calibrated':True,'units':'metres',
                 'map_to_local_m':{'matrix':[[2,0],[0,2]],'offset':[1,-1]}}
    package=write_untrained_package(build_clf_model(len(grid['cells'])),
                                    Path(directory)/'package',contract,grid,calibration)
    runtime=ModelRuntime(package);runtime.load()
    assert runtime.status()['state']=='incompatible_model'
    assert 'status=trained' in runtime.status()['error']
    # Explicit fixture attestation for loader testing only, never a real export.
    path=package/'model_metadata.json';metadata=json.loads(path.read_text())
    metadata['status']='trained'
    metadata['training_metadata']={'synthetic_software_fixture':True,'weights_are_untrained':True}
    path.write_text(json.dumps(metadata))
    runtime=ModelRuntime(package);runtime.load()
    assert runtime.status()['state']=='ready',runtime.status()
    pipeline=FeaturePipeline(package/'feature_contract.json',package/'preprocessing.json')
    packet=SensorPacket.model_validate({'schema_version':'1.0','packet_id':str(uuid4()),
        'session_id':str(uuid4()),'device_id':'software-test','timestamp_ms':1,
        'wifi':[{'bssid':'02:00:00:00:00:00','rssi_dbm':-65}]})
    result=runtime.predict(pipeline.process(packet))
    assert result['location'] is None and result['location_status']=='candidates_only'
    assert len(result['location_candidates'])==3
    assert all(np.isfinite([x['x'],x['y'],x['score']]).all() for x in result['location_candidates'])
    print('PASS: real .keras loading, 24-AP single input, candidate decoding; untrained default rejected')
    print('Synthetic software verification only; no trained model or localization accuracy claimed.')
