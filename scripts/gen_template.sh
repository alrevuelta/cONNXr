#!/bin/bash

DOMAIN=$1
OPERATOR=$2
VERSION=$3

ROOT=$(git rev-parse --show-toplevel)
cd $ROOT/scripts

if [[ -z $DOMAIN ]]
then
    echo "usage: $0 [domain] [operator] [version]"
    echo "possible domains:"
    python3 -m onnx_generator \
        --list-domains \
        -- ..
    exit 1
fi

if [[ -z $OPERATOR ]]
then
    echo "usage: $0 [domain] [operator] [version]"
    echo "possible operators:"
    if ! python3 -m onnx_generator \
        --list-operators \
        --domains $DOMAIN \
        -- ..
    then
        echo "no operators found! check specified domain!"
    fi
    exit 1
fi

if [[ -z $VERSION ]]
then
    echo "usage: $0 [domain] [operator] [version]"
    echo "possible versions:"
    if ! python3 -m onnx_generator \
        --list-versions \
        --domains $DOMAIN \
        -i '^'$OPERATOR'$' \
        -- ..
    then
        echo "no versions found! check specified domain and operator!"
    fi
    exit 1
fi

RESULT=$(python3 -m onnx_generator \
    --list-versions \
    --domains $DOMAIN \
    -i '^'$OPERATOR'$' \
    --version $VERSION \
    -- ..)
if [[ $RESULT != $@ ]]
then
    echo "could not find combination of specified domain, operator and version!"
    exit 1
fi

echo "### GENERATING TEMPLATE FOR $@ ###"
echo
python3 -m onnx_generator -vv \
    --domains $DOMAIN \
    -i '^'$OPERATOR'$' \
    --version $VERSION \
    -- ..
echo

OPERATORS=
echo "### GENERATING NEW OPERATOR SET ###"
echo
OPERATORS=
for domain in $(python3 -m onnx_generator --list-domains -- ..)
do
    if [[ -d ../src/operators/$domain ]]
    then
        OPERATORS+=$(ls ../src/operators/$domain/ | awk '{print "^"$0"$"}')
    fi
done
python3 -m onnx_generator -v \
    -i $OPERATORS\
    --version latest \
    --force-pattern \
        '^'$ROOT'/src/operators/'$DOMAIN'/opdomain_.*$' \
        '^'$ROOT'/src/operators/operator_set.c$' \
    --skip-pattern '.*' \
    -- ..
