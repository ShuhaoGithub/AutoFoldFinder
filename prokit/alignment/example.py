import pytest
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config"))
import celery_config

from requests_toolbelt import MultipartEncoder
import requests

hostip = celery_config.HOSTNAME
callbackPort = celery_config.CALLBACK_PORT
keyPort =celery_config.KEY_PORT
testPort = celery_config.APITEST_PORT

url1 = f"http://{hostip}:{testPort}/w1-api/v1/to-splited-mol2"
url2 = f"http://{hostip}:{testPort}/w1-api/v1/cavity" 


class TestCavityAPI(object):
    def test_case_1(self):
        """mode:1, modelig:1, ligandfile: filename.sdf 
        """
        cookie =test_getcookie()
        filename = "data/1db4_lig.sdf"
        jobid = "200841"
        with open(filename, "rb") as f:
            m = MultipartEncoder(
                fields={
                    "jobid": jobid,
                    "pdbpath": f"1/{jobid}/generation/1db4.pdb",
                    "chainid": "A",
                    "mode": "1",
                    "modelig": "1",
                    "ligandfile": (os.path.basename(filename), f, "application/octet-stream"),
                    "modefrag": "0",
                }
            )
            r = requests.post(url2, data=m, headers={"Content-Type": m.content_type, "Cookie": cookie})
            assert r.status_code == 200
            print(r.json())
            

    def test_case_2(self):
        """mode:1, modelig:1, ligandfile: filename.mol
        """
        cookie =test_getcookie()
        filename = "data/1db4_lig.mol"
        jobid = "200841"
        with open(filename, "rb") as f:
            m = MultipartEncoder(
                fields={
                    "jobid": jobid,
                    "pdbpath": f"1/{jobid}/generation/1db4.pdb",
                    "chainid": "A",
                    "mode": "1",
                    "modelig": "1",
                    "ligandfile": (os.path.basename(filename), f, "application/octet-stream"),
                    "modefrag": "0",
                }
            )
            r = requests.post(url2, data=m, headers={"Content-Type": m.content_type, "Cookie": cookie})
            assert r.status_code == 200
            assert r.json()


class TestMolProcess(object):
    def test_sdf2mol2_case1(self):
        url = f"http://{hostip}:{testPort}/w1-api/v1/sdf-to-mol2"
        m = MultipartEncoder(
            fields={
                "uploadmolpath": "1/mockdata/evaluation/up3d_demo_plk1_top9_new.sdf", 
            })
        r = requests.post(url, data=m, headers={"Content-Type": m.content_type})
        assert r.status_code == 200
        print(url)
        print(r.text)
        return r.text

    def test_sdf2mol2_case2(self):
        url = f"http://{hostip}:{testPort}/w1-api/v1/sdf-to-mol2"
        m = MultipartEncoder(
            fields={
                "uploadmolpath": "4/200395/molexplorer/splitedmols/3d/up3d_upload_1.sdf", 
            })
        r = requests.post(url, data=m, headers={"Content-Type": m.content_type})
        assert r.status_code == 200
        print(url)
        print(r.text)
        return r.text

    def test_split_mol_case1(self):
        m = MultipartEncoder(
            fields={
                "uploadmolpath": "1/mockdata/evaluation/up3d_demo_plk1_top9_new.sdf",
            })
        r = requests.post(url1, data=m, headers={"Content-Type": m.content_type})
        assert r.status_code == 200
        print(url1)
        print(r.text)
        return r.text

    def test_split_mol_case2(self):
        m = MultipartEncoder(
            fields={
                "uploadmolpath": "1/mockdata/evaluation/up2d_hs-design-1.smi",
            })
        r = requests.post(url1, data=m, headers={"Content-Type": m.content_type})
        assert r.status_code == 200
        print(url1)
        print(r.text)
        return r.text

    
    def test_split_mol_case3(self):
        m = MultipartEncoder(
            fields={
                "uploadmolpath": "2/10158/evaluation/rerun0000001/up3d_6kx7-dockout-XP-1.sdf",
            })
        r = requests.post(url1, data=m, headers={"Content-Type": m.content_type})
        assert r.status_code == 200
        print(url1)
        print(r.text)
        return r.text

    def test_ligand_in_cavity_case1(self):
        url1= f"http://{hostip}:{testPort}/w1-api/v1/check-ligand-in-cavity"
        m = MultipartEncoder(
            fields={
                "ligandpath": "2/debug1/optimization/rerun001/ligand.mol2",
                "cavitypath": "2/debug1/optimization/rerun001/cavity-output/2owb-A_cavity_1.pdb",
            })
        r = requests.post(url1, data=m, headers={"Content-Type": m.content_type})
        assert r.status_code == 200
        print(url1)
        print(r.text)
        return r.text

    def test_ligand_in_cavity_case2(self):
        url1= f"http://{hostip}:{testPort}/w1-api/v1/check-ligand-in-cavity"
        m = MultipartEncoder(
            fields={
                "ligandpath": "2/10247/evaluation/rerun0000001/splitedmols/3d/up3d_6kx7-dockout-XP-1_99.mol2",
                "cavitypath": "2/debug1/optimization/rerun001/cavity-output/2owb-A_cavity_1.pdb",
            })
        r = requests.post(url1, data=m, headers={"Content-Type": m.content_type})
        assert r.status_code == 200
        print(url1)
        print(r.text)
        return r.text

def test_getcookie():
    url = f"http://{hostip}:{keyPort}/login/"
    print(url)
    data = {"LoginName": "raiden", "Password": "asdf"}
    res = requests.post(url=url, data=data)
    cookie = res.request.headers.get("Cookie")
    assert cookie is not None
    return cookie


