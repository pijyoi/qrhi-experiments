.SUFFIXES: .qsb

objects = texture.vert.qsb texture.frag.qsb rescale.comp.qsb

shading_langs = --glsl "310 es, 430" --hlsl 50 --msl 12

all: $(objects)

ifdef MAKEDIR:
!ifdef MAKEDIR

# nmake
RM = del

$(objects): $*
	pyside6-qsb $(shading_langs) -o $@ $*

!else
else

# gnu make
RM = rm

$(objects): %.qsb: %
	pyside6-qsb $(shading_langs) -o $@ $<

endif
!endif :

clean:
	$(RM) *.qsb
