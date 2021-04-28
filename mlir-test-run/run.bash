export BDIR=/home/research/Documents/PNNL/llvm-build/bin
export LDIR=/home/research/Documents/PNNL/llvm-install/lib

INPUT=$1

# Check if mlir-cpu-runner printing libraries exist
if [ -f "$LDIR/libmlir_runner_utils.so" ]; then
    : # do nothing
else
    echo "Could not find compiled mlir-cpu-runner printing libraries."
    echo "  Verify that LDIR is set to the correct path in the run.sh file"
    echo "  Current value: $LDIR"
    exit 1
fi

# Check if mlir-opt printing libraries exist
if [ -f "$BDIR/mlir-opt" ]; then
    : # do nothing
else
    echo "Could not find mlir-opt."
    echo "  Verify that BDIR is set to the correct path in the run.sh file"
    echo "  Current value: $BDIR"
    exit 1
fi


# Check if mlir-opt printing libraries exist
if [ -f "$1" ]; then
    : # do nothing
else
    echo "Could not find the input mlir file: $1"
    echo "  Is the filepath correct?"
    exit 1
fi


# Stop execution on first error and verbose print the executed commands
# set -e
# set -x
# $BDIR/mlir-opt $INPUT -linalg-generalize-named-ops -o tmp.mlir
$BDIR/mlir-opt $INPUT -o tmp.mlir

# $BDIR/mlir-opt tmp.mlir -linalg-bufferize -convert-linalg-to-loops -o tmp1.mlir
$BDIR/mlir-opt tmp.mlir -linalg-bufferize -convert-linalg-to-affine-loops -o tmp1.mlir
# $BDIR/mlir-opt tmp.mlir -linalg-bufferize -convert-linalg-to-affine-loops --affine-loop-unroll -o tmp1.mlir
# $BDIR/mlir-opt tmp.mlir -linalg-bufferize -convert-linalg-to-llvm -o tmp1.mlir

$BDIR/mlir-opt tmp1.mlir --func-bufferize --tensor-constant-bufferize --tensor-bufferize --finalizing-bufferize -o tmp2.mlir
$BDIR/mlir-opt tmp2.mlir -lower-affine -convert-scf-to-std  -convert-std-to-llvm -o tmp3.mlir
$BDIR/mlir-cpu-runner tmp3.mlir -e main -entry-point-result=void -shared-libs=$LDIR/libmlir_runner_utils.so,$LDIR/libmlir_c_runner_utils.so

rm tmp*
#set +x
