HUMIXMODULE=lib/HumixFaceRec.node

all: $(HUMIXMODULE)
$(HUMIXMODULE): src/HumixFaceRec.cpp
	node-gyp configure
	node-gyp build
	

clean:
	rm -rf build $(HUMIXMODULE)
