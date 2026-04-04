.PHONY: fmt lint test build

# Binary name
TAG ?= 0.2.0
BINARY_NAME=bin/server
MAIN_PATH=cmd/server/main.go

all: lint test build

build: ## Compile the binary
	go build -o $(BINARY_NAME) $(MAIN_PATH)

test: ## Run unit tests
	go test -v ./...

fmt: ## Format the code
	go fmt ./...

vet: ## Run go vet
	go vet ./...

lint: ## Run custom-gcl if installed
	@if command -v custom-gcl > /dev/null; then \
		custom-gcl run; \
	else \
		echo "custom-gcl not installed, skipping..."; \
	fi

example: ## Run example
	go run cmd/example/main.go
