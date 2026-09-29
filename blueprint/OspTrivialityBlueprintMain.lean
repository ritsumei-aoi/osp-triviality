import VersoManual
import VersoBlueprint.PreviewManifest
import OspTrivialityBlueprint.Blueprint

open Verso Doc
open Verso.Genre Manual

def main (args : List String) : IO UInt32 :=
  Informal.PreviewManifest.blueprintMainWithPreviewData
    (%doc OspTrivialityBlueprint.Blueprint)
    args
    (extensionImpls := by exact extension_impls%)
