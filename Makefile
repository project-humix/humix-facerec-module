HUMIXMODULE=lib/HumixFaceRec.node

all: $(HUMIXMODULE)
$(HUMIXMODULE): src/HumixFacerec.cpp
	node-gyp configure
	node-gyp build
	

clean:
	rm -rf build $(HUMIXMODULE)
