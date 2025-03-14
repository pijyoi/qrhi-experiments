.SUFFIXES: .qsb

objects = shaded.vert.qsb shaded.frag.qsb

all: $(objects)

$(objects): $*
	pyside6-qsb --qt6 -o $@ $*
