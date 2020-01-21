include(VersionAndGitRef)
set_version(1.0.0)
get_gitref()

set(version ${PURIPSI_VERSION})
string(REGEX REPLACE "\\." ";" version "${PURIPSI_VERSION}")
list(GET version 0 PURIPSI_VERSION_MAJOR)
list(GET version 1 PURIPSI_VERSION_MINOR)
list(GET version 2 PURIPSI_VERSION_PATCH)

