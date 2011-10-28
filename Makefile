IFLAGS = `pkg-config --cflags opencv` -O3
LFLAGS = `pkg-config --libs opencv`

all: congealReal funnelReal

clean:
	rm congealReal funnelReal

congealReal: congealReal.cpp
	gcc $(IFLAGS) $(LFLAGS) -o congealReal congealReal.cpp

funnelReal: funnelReal.cpp
	gcc $(IFLAGS) $(LFLAGS) -o funnelReal funnelReal.cpp
