FLAGS=-O3 -I/usr1/home/rbalasub/ResearchTools/dlib-17.43 

all: dirs bin/link_lda

dirs:
	@if [ ! -d bin ]; then mkdir bin; fi; if [ ! -d lib ]; then mkdir lib; fi

bin/link_lda: lib/link_lda.o lib/config.o lib/options.o lib/util.o lib/links.o lib/corpus.o lib/model.o lib/stats.o lib/hungarian.o
	g++ $(FLAGS) lib/link_lda.o lib/config.o lib/options.o lib/util.o  lib/links.o lib/model.o lib/corpus.o lib/stats.o lib/hungarian.o  -o bin/link_lda

lib/link_lda.o: src/cpp/link_lda.cpp src/cpp/util.h src/cpp/config.h
	g++ -c $(FLAGS) src/cpp/link_lda.cpp -o lib/link_lda.o

lib/config.o: src/cpp/config.cpp src/cpp/config.h src/cpp/util.h
	g++ -c $(FLAGS) src/cpp/config.cpp -o lib/config.o

lib/options.o: src/cpp/options.cpp src/cpp/options.h
	g++ -c $(FLAGS) src/cpp/options.cpp -o lib/options.o

lib/util.o: src/cpp/util.cpp src/cpp/util.h
	g++ -c $(FLAGS) src/cpp/util.cpp -o lib/util.o

lib/links.o: src/cpp/links.cpp src/cpp/links.h src/cpp/util.h src/cpp/config.h
	g++ -c $(FLAGS) src/cpp/links.cpp -o lib/links.o

lib/model.o: src/cpp/model.cpp src/cpp/model.h src/cpp/util.h src/cpp/component.h src/cpp/config.h
	g++ -c $(FLAGS) src/cpp/model.cpp -o lib/model.o

lib/corpus.o: src/cpp/corpus.cpp src/cpp/corpus.h src/cpp/util.h src/cpp/component.h src/cpp/config.h
	g++ -c $(FLAGS) src/cpp/corpus.cpp -o lib/corpus.o

lib/stats.o: src/cpp/stats.cpp src/cpp/stats.h
	g++ -c $(FLAGS) src/cpp/stats.cpp -o lib/stats.o

lib/hungarian.o: src/cpp/hungarian.c src/cpp/hungarian.h
	g++ -c $(FLAGS) src/cpp/hungarian.c -o lib/hungarian.o

clean:
	rm bin/link_lda lib/link_lda.o lib/config.o lib/options.o lib/links.o lib/model.o lib/corpus.o lib/stats.o lib/util.o lib/hungarian.o

hdp: src/cpp/hdp.cpp
	g++ -O3 src/cpp/hdp.cpp -lgsl  -lgslcblas -o bin/hdp
