# ShewhartConstants

Shewhart Constants

Try Follow The Table of Control Chart Constants

| n   | d2    | d3     | c4     | A2    | D3    | D4    | A3    | B3    | B4    |
| --- | ----- | ------ |--------| ----- | ----- | ----- | ----- | ----- | ----- |
| 2   | 1.128 | 0.8525 | 0.7979 | 1.880 | 0     | 3.267 | 2.659 | 0     | 3.267 |
| 3   | 1.693 | 0.8884 | 0.8862 | 1.023 | 0     | 2.574 | 1.954 | 0     | 2.568 |
| 4   | 2.059 | 0.8798 | 0.9213 | 0.729 | 0     | 2.282 | 1.628 | 0     | 2.266 |
| 5   | 2.326 | 0.8798 | 0.9400 | 0.577 | 0     | 2.114 | 1.427 | 0     | 2.089 |
| 6   | 2.534 | 0.8480 | 0.9515 | 0.483 | 0     | 2.004 | 1.287 | 0.030 | 1.970 |
| 7   | 2.704 | 0.8332 | 0.9594 | 0.419 | 0.076 | 1.924 | 1.182 | 0.118 | 1.882 |
| 8   | 2.847 | 0.8198 | 0.9650 | 0.373 | 0.136 | 1.864 | 1.099 | 0.185 | 1.815 |
| 9   | 2.970 | 0.8078 | 0.9693 | 0.337 | 0.184 | 1.816 | 1.032 | 0.239 | 1.761 |
| 10  | 3.078 | 0.7971 | 0.9727 | 0.308 | 0.223 | 1.777 | 0.975 | 0.284 | 1.716 |
| 11  | 3.173 | 0.7873 | 0.9754 | 0.285 | 0.256 | 1.744 | 0.927 | 0.321 | 1.679 |
| 12  | 3.258 | 0.7785 | 0.9776 | 0.266 | 0.283 | 1.717 | 0.886 | 0.354 | 1.646 |
| 13  | 3.336 | 0.7704 | 0.9794 | 0.249 | 0.307 | 1.693 | 0.850 | 0.382 | 1.618 |
| 14  | 3.407 | 0.7630 | 0.9810 | 0.235 | 0.328 | 1.672 | 0.817 | 0.406 | 1.594 |
| 15  | 3.472 | 0.7562 | 0.9823 | 0.223 | 0.347 | 1.653 | 0.789 | 0.428 | 1.572 |
| 16  | 3.532 | 0.7499 | 0.9835 | 0.212 | 0.363 | 1.637 | 0.763 | 0.448 | 1.552 |
| 17  | 3.588 | 0.7441 | 0.9845 | 0.203 | 0.378 | 1.662 | 0.739 | 0.466 | 1.534 |
| 18  | 3.640 | 0.7386 | 0.9854 | 0.194 | 0.391 | 1.607 | 0.718 | 0.482 | 1.518 |
| 19  | 3.689 | 0.7335 | 0.9862 | 0.187 | 0.403 | 1.597 | 0.698 | 0.497 | 1.503 |
| 20  | 3.735 | 0.7287 | 0.9869 | 0.180 | 0.415 | 1.585 | 0.680 | 0.510 | 1.490 |
| 21  | 3.778 | 0.7272 | 0.9876 | 0.173 | 0.425 | 1.575 | 0.663 | 0.523 | 1.477 |
| 22  | 3.819 | 0.7199 | 0.9882 | 0.167 | 0.434 | 1.566 | 0.647 | 0.534 | 1.466 |
| 23  | 3.858 | 0.1759 | 0.9887 | 0.162 | 0.443 | 1.557 | 0.633 | 0.545 | 1.455 |
| 24  | 3.895 | 0.7121 | 0.9892 | 0.157 | 0.451 | 1.548 | 0.619 | 0.555 | 1.445 |
| 25  | 3.931 | 0.7084 | 0.9896 | 0.153 | 0.459 | 1.541 | 0.606 | 0.565 | 1.435 |

... And N>=26 value

```python
sc = ShewhartConstants() # or ShewhartConstantsFix() 
# ShewhartConstantsFix() Fix the math gamma limited by float 64 bit range Problem
# https://stackoverflow.com/questions/64490723/math-gamma-limited-by-float-64-bit-range-any-way-to-assign-more-bits

n = 2

sc.d2(n)

sc.d3(n)

sc.c4(n)

sc.A2(n)

sc.A3(n)

sc.D3(n)

sc.D4(n)

sc.B3(n)

sc.B4(n)
```