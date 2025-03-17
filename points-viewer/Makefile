.SUFFIXES: .qsb

objects = points.vert.qsb points.frag.qsb

shading_langs = --glsl "300 es, 140" --hlsl 50 --msl 12

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
