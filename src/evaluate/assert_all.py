from .assert_dt import main as main1
from .assert_dt2 import main as main2
from .assert_nn import main as main3
from .assert_nn2 import main as main4

def run_evaluation():
    print('Asserting performance of decision tree through overfitting iris data')
    main1()
    print()

    print('Asserting performance of decision tree through plotting 2d-decision regions for toy data')
    main2()
    print()

    print('Asserting performance of neural net through overfitting iris data')
    main3()
    print()

    print('Asserting performance of neural net through plotting 2d-decision regions for toy data')
    main4()
    print()

if __name__ == '__main__':
    run_evaluation()
