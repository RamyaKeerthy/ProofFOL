import re
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from .fol_prover9_parser import Prover9_FOL_Formula
from .Formula import FOL_Formula
import re
import os

# from Formula import FOL_Formula

# set the path to the prover9 executable
os.environ['PROVER9'] = '/Prover9/bin'


class FOL_Prover9_Program:
    def __init__(self, logic_program: str, logic_premises, logic_conclusion, dataset_name='FOLIO') -> None:
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        self.logic_premises = logic_premises
        self.logic_conclusion = logic_conclusion
        self.flag = self.parse_logic_program()

    def parse_logic_program(self):
        try:
            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True
        except:
            return False

    def execute_program(self):
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 10
            # prover = Prover9()
            # result = prover.prove(goal, assumptions)

            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()  # fails to run on Mac
            # print(prover.proof())
            if result:
                return 'True', ''
            else:
                # If Prover9 fails to prove, we differentiate between False and Unknown
                # by running Prover9 with the negation of the goal
                negated_goal = NegatedExpression(goal)
                # negation_result = prover.prove(negated_goal, assumptions)
                prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
                negation_result = prover.prove()
                if negation_result:
                    return 'False', ''
                else:
                    return 'Unknown', ''
        except Exception as e:
            print(self.prover9_conclusion)
            print(self.prover9_premises)
            return None, str(e)

    def answer_mapping(self, answer):
        if answer == 'True':
            return 'A'
        elif answer == 'False':
            return 'B'
        elif answer == 'Unknown':
            return 'C'
        else:
            raise Exception("Answer not recognized")


if __name__ == "__main__":
    # ground-truth: True
    logic_program = """Premises:
    Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
    ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    ∃y ∃x (Czech(x) ∧ Author(x, y) ∧ Book(y) ∧ Publish(y, year1946)) ::: A Czech person wrote a book in 1946.
    """

    prover9_program = FOL_Prover9_Program(logic_program)
    answer, error_message = prover9_program.execute_program()
    print('Program:', answer)