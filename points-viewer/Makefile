.SUFFIXES: .qsb

objects = points.vert.qsb points.frag.qsb

all: $(objects)

$(objects): $*
	pyside6-qsb --qt6 -o $@ $*
