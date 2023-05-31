CC=gcc
CFLAGS=-Wall -fPIC -c
EXECUTABLE=ina.so
SOURCES=ina.c run.c
OBJECTS=$(SOURCES:.c=.o)

all:clean o $(EXECUTABLE)
$(EXECUTABLE):$(OBJECTS)
	$(CC) -shared $(OBJECTS) -o $(EXECUTABLE) -lm  -L./lib -lsundials_cvode -Wl,-rpath='$$ORIGIN/lib'

o:ina.c
	$(CC) $(CFLAGS) -I./include $(SOURCES) $(SOURCES)

clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)
