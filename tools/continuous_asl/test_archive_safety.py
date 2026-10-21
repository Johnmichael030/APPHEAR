"""Security regressions use tiny generated fixtures retained only under the experiment."""
import argparse
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
import zipfile
import archive_safety as safety
ROOT = None

class ArchiveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.case = Path(tempfile.mkdtemp(prefix="archive_case_", dir=ROOT))
    def makezip(self, names):
        path=self.case/"fixture.zip"
        with zipfile.ZipFile(path,"w") as archive:
            for name in names:
                archive.writestr(name,b"abc")
        return path
    def test_valid_zip_checks_and_extracts_without_deleting_source(self):
        archive=self.makezip(["folder/a.txt"])
        target=self.case/"new"
        result=safety.extract(archive,target,self.case,100)
        self.assertEqual(result["files"],1)
        self.assertEqual((target/"folder/a.txt").read_bytes(),b"abc")
        self.assertTrue(archive.exists())
        with self.assertRaises(FileExistsError):
            safety.extract(archive,target,self.case,100)
    def test_parent_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"UNSAFE_ARCHIVE_PATH"):
            safety.inspect_archive(self.makezip(["../escape.txt"]),100)
        self.assertFalse((self.case.parent/"escape.txt").exists())
    def test_windows_absolute_ads_and_device_paths_are_rejected(self):
        for name in ["C:/evil","folder/file:stream","/absolute",r"\\server\share","NUL.txt","folder/x."]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                safety.member_path(name)
    def test_case_insensitive_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"DUPLICATE"):
            safety.inspect_archive(self.makezip(["A.txt","a.txt"]),100)
    def test_budget_rejected_before_extract(self):
        with self.assertRaisesRegex(ValueError,"BUDGET"):
            safety.inspect_archive(self.makezip(["a","b"]),5)
    def test_zip_symlink_rejected(self):
        path=self.case/"link.zip"
        info=zipfile.ZipInfo("link")
        info.create_system=3
        info.external_attr=(stat.S_IFLNK|0o777)<<16
        with zipfile.ZipFile(path,"w") as archive:
            archive.writestr(info,"../outside")
        with self.assertRaisesRegex(ValueError,"LINK"):
            safety.inspect_archive(path,100)
    def test_tar_hardlink_rejected(self):
        path=self.case/"link.tar"
        with tarfile.open(path,"w") as archive:
            info=tarfile.TarInfo("link")
            info.type=tarfile.LNKTYPE
            info.linkname="../outside"
            archive.addfile(info)
        with self.assertRaisesRegex(ValueError,"LINK"):
            safety.inspect_archive(path,100)
    def test_output_must_stay_inside_root(self):
        archive=self.makezip(["a"])
        with self.assertRaisesRegex(ValueError,"ESCAPES"):
            safety.extract(archive,self.case.parent/"outside",self.case,100)
    def test_source_output_nested_or_identical_rejected(self):
        src=self.case/"source"
        for target in [src,src/"nested",self.case]:
            with self.subTest(target=target),self.assertRaisesRegex(ValueError,"OVERLAP"):
                safety.assert_disjoint([src],target)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixtures-root",required=True,type=Path)
    args,rest=parser.parse_known_args()
    ROOT=args.fixtures_root.resolve()
    if "experiments" not in ROOT.parts or "regressions" not in ROOT.parts:
        raise SystemExit("Fixtures must stay in the experiment regressions directory")
    ROOT.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ArchiveSafetyTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

