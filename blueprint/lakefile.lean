import Lake
open Lake DSL

-- The blueprint is a separate package: it requires the formalization by path and never edits it.
require VersoBlueprint from git "https://github.com/leanprover/verso-blueprint"@"v4.34.0"
require InhomogeneousDeformations from ".."

package OspTrivialityBlueprint where
  precompileModules := false
  leanOptions := #[⟨`experimental.module, true⟩]

@[default_target]
lean_lib OspTrivialityBlueprint where
