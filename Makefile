all: solve_system

solve_system: main.o matrix_io.o gaussian_elimination.o functions.o
	g++ main.o matrix_io.o gaussian_elimination.o functions.o -o solve_system

main.o: main.cpp matrix_io.h gaussian_elimination.h functions.h
	g++ -g -Wall -Wextra -Werror -pedantic -std=gnu++0x -O3 -c main.cpp

matrix_io.o: matrix_io.cpp matrix_io.h
	g++ -g -Wall -Wextra -Werror -pedantic -std=gnu++0x -O3 -c matrix_io.cpp

gaussian_elimination.o: gaussian_elimination.cpp gaussian_elimination.h
	g++ -g -Wall -Wextra -Werror -pedantic -std=gnu++0x -O3 -c gaussian_elimination.cpp

functions.o: functions.cpp functions.h
	g++ -g -Wall -Wextra -Werror -pedantic -std=gnu++0x -O3 -c functions.cpp

clean:
	rm -f *.o solve_system
