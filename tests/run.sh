#! /bin/bash

FAILED=0

for f in *.x; do
  echo "Running test $f"
  OMP_NUM_THREADS=1 ./$f
  if [ $? -ne 0 ]; then
    FAILED=$((FAILED + 1))
    echo "-> failed"
  else
    echo "-> passed"
  fi
done

echo
if [ $FAILED -gt 0 ]; then
  echo "$FAILED TEST(S) FAILED!"
else
  echo "All tests passed!"
fi
