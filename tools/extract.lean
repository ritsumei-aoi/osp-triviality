import InhomogeneousDeformations
open Lean

/-- ours: in the project namespace -/
def ours (n : Name) : Bool := (`InhomogeneousDeformations).isPrefixOf n

/-- a "real" node: ours, and not a compiler-generated detail -/
def isNode (n : Name) : Bool := ours n && !n.isInternalDetail && !n.isInternal

/-- BFS from a declaration, passing *through* internal details, collecting the
    real declarations first reached. -/
partial def realDeps (env : Environment) (start : Name) : NameSet := Id.run do
  let mut seen : NameSet := {}
  let mut acc  : NameSet := {}
  let mut work : List Name := [start]
  while !work.isEmpty do
    let n := work.head!
    work := work.tail!
    if seen.contains n then continue
    seen := seen.insert n
    match env.find? n with
    | none => continue
    | some ci =>
      for d in ci.getUsedConstantsAsSet do
        if !ours d then continue
        if d == start then continue
        if isNode d then
          acc := acc.insert d
        else if !seen.contains d then
          work := d :: work
  return acc

def kindOf : ConstantInfo → String
  | .axiomInfo _ => "axiom"
  | .defnInfo _  => "def"
  | .thmInfo _   => "theorem"
  | .opaqueInfo _=> "opaque"
  | .quotInfo _  => "quot"
  | .inductInfo _=> "inductive"
  | .ctorInfo _  => "ctor"
  | .recInfo _   => "rec"

#eval show CoreM Unit from do
  let env ← getEnv
  let mut nodes : Array String := #[]
  let mut edges : Array String := #[]
  let mut names : Array Name := #[]
  for (n, ci) in env.constants.toList do
    if isNode n then
      names := names.push n
      let m := match env.getModuleFor? n with | some mn => mn.toString | none => "?"
      nodes := nodes.push s!"{n}\t{m}\t{kindOf ci}"
  for n in names do
    for d in realDeps env n do
      edges := edges.push s!"{n}\t{d}"
  IO.FS.writeFile "nodes.tsv" (String.intercalate "\n" nodes.toList ++ "\n")
  IO.FS.writeFile "edges.tsv" (String.intercalate "\n" edges.toList ++ "\n")
  IO.println s!"nodes {nodes.size}  edges {edges.size}"
