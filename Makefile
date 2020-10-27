CC=mpicc
CFLAGS=-Wall -Wextra -Iinclude -std=c99 -O3 -g -fopenmp
LDFLAGS=-L. -lm -lcphis -fopenmp

# Uncomment to enable the Tpetra (Trilinos) linear algebra backend.
# Tests and examples cannot be compiled when this is enabled!
#CFLAGS+=-Ibackends/tpetra -DCPHIS_HAVE_TPETRA

default: libcphis.a

all: libcphis.a tests examples doc

libcphis.a: $(patsubst src/%.c, src/%.o, $(wildcard src/*.c))
	ar -rs libcphis.a $^

tests: libcphis.a $(patsubst tests/%.c, tests/%.x, $(wildcard tests/*.c))
	(cd tests; ./run.sh; cd ..)

examples: libcphis.a \
$(patsubst examples/%.c, examples/%.x, $(wildcard examples/*.c))

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

tests/%.x: tests/%.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lm

examples/%.x: examples/%.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lm

doc: doxygen.conf
	doxygen doxygen.conf

clean:
	rm -f src/*.o
	rm -f tests/*.x
	rm -f examples/*.x
	rm -f libcphis.a
	rm -rf doc

.PHONY: clean doc
