CC = gcc
CFLAGS = -O2 -I./src -std=c11
SRC = src/tokenizer.c
BIN_DIR = bin

.PHONY: all test clean
all: $(BIN_DIR)/test_tokenizer

$(BIN_DIR)/test_tokenizer: src/test_tokenizer.c $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ src/test_tokenizer.c $(SRC)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

test: all
	$(BIN_DIR)/test_tokenizer

clean:
	rm -f $(BIN_DIR)/test_tokenizer
