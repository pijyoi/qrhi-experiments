SUBDIRS = mesh-viewer image-viewer points-viewer image-rescale image-rescale-compute

all:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir; done
