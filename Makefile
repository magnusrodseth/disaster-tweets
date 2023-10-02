
.PHONY: all assignment-2

all: assignment-2

assignment-2: 
	cd assignment-2 && pandoc -V geometry:margin=1in -o ASSIGNMENT_2.pdf README.md