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

lint: ## Run golangci-lint if installed
	@if command -v golangci-lint > /dev/null; then \
		golangci-lint run; \
	else \
		echo "golangci-lint not installed, skipping..."; \
	fi

example: ## Run example
	go run cmd/example/main.go
