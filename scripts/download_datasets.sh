#!/bin/bash

# Create data directory
mkdir -p data/sparql_datasets

echo "=== Downloading NL-to-SPARQL Datasets ==="

# LC-QuAD 2.0
echo "1. LC-QuAD 2.0"
echo "   Visit: http://lc-quad.sda.tech/lcquad2.0.html"
echo "   Download train.json and test.json"
mkdir -p data/sparql_datasets/lcquad2
echo "   Place files in: data/sparql_datasets/lcquad2/"
echo ""

# QALD-9
echo "2. QALD-9"
echo "   Visit: https://github.com/ag-sc/QALD/tree/master/9/data"
echo "   Download qald-9-train-multilingual.json"
mkdir -p data/sparql_datasets/qald9
echo "   Place files in: data/sparql_datasets/qald9/"
echo ""

echo "=== Download Instructions Displayed ==="
echo "After downloading, run: python src/main/fine_tune_t5_sparql.py"
