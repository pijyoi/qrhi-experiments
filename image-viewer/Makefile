.SUFFIXES: .qsb

objects = texture.vert.qsb texture.frag.qsb

shading_langs = --glsl "300 es,140" --hlsl 50 --msl 12

all: $(objects)

$(objects): $*
	pyside6-qsb $(shading_langs) -o $@ $*
