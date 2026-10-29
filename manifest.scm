;; Guix development environment for the research lab.
;; Guix supplies the toolchain; PyPI packages (langchain, langgraph, ...) are locked in uv.lock and
;; installed by uv into .venv, because Guix does not package the LangChain stack.
;; Enter with direnv (.envrc), or from bash: source dev/env.sh
(specifications->manifest
 '("python@3.12"
   "uv"
   "gcc-toolchain"   ; libgcc_s for manylinux wheels (pydantic-core, jiter, ormsgpack, primp, uuid-utils)
   "sqlite"          ; sqlite3 CLI to inspect the datarooms (data/*.db)
   "git"
   "coreutils"
   "bash"))
