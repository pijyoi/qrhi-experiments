.SUFFIXES: .qsb

objects = texture.vert.qsb texture.frag.qsb

all: $(objects)

$(objects): $*
	pyside6-qsb --qt6 -o $@ $*
