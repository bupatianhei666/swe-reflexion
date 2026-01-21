import libcst as cst
import libcst.matchers as m


class CompressTransformer(cst.CSTTransformer):
    DESCRIPTION = str = "Replaces function body with ..."
    replacement_string = '"$$FUNC_BODY_REPLACEMENT_STRING$$"'

    def __init__(self, keep_constant=True):
        self.keep_constant = keep_constant

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        new_body = [
            stmt
            for stmt in updated_node.body
            if m.matches(stmt, m.ClassDef())
            or m.matches(stmt, m.FunctionDef())
            or (
                self.keep_constant
                and m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Assign())
            )
        ]
        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        # Remove docstring in the class body
        new_body = [
            stmt
            for stmt in updated_node.body.body
            if not (
                m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Expr())
                and m.matches(stmt.body[0].value, m.SimpleString())
            )
        ]
        return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        new_expr = cst.Expr(value=cst.SimpleString(value=self.replacement_string))
        new_body = cst.IndentedBlock((new_expr,))
        # another way: replace with pass?
        return updated_node.with_changes(body=new_body)


code = """
def inverse_trig(rubi):
    pattern1 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))))
    rule1 = ReplacementRule(pattern1, lambda b, c, n, a, x : -b*c*n*Int(x*(a + b*asin(c*x))**(n + S(-1))/sqrt(-c**S(2)*x**S(2) + S(1)), x) + x*(a + b*asin(c*x))**n)
    rubi.add(rule1)
    
    pattern2 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))))
    rule2 = ReplacementRule(pattern2, lambda b, c, n, a, x : b*c*n*Int(x*(a + b*acos(c*x))**(n + S(-1))/sqrt(-c**S(2)*x**S(2) + S(1)), x) + x*(a + b*acos(c*x))**n)
    rubi.add(rule2)
    
    pattern3 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))))
    rule3 = ReplacementRule(pattern3, lambda b, c, n, a, x : c*Int(x*(a + b*asin(c*x))**(n + S(1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(b*(n + S(1))) + (a + b*asin(c*x))**(n + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule3)
    
    pattern4 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))))
    rule4 = ReplacementRule(pattern4, lambda b, c, n, a, x : -c*Int(x*(a + b*acos(c*x))**(n + S(1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(b*(n + S(1))) - (a + b*acos(c*x))**(n + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule4)

    pattern5 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n, x: FreeQ(n, x)))
    rule5 = ReplacementRule(pattern5, lambda b, c, n, a, x : Subst(Int(x**n*cos(a/b - x/b), x), x, a + b*asin(c*x))/(b*c))
    rubi.add(rule5)

    pattern6 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n, x: FreeQ(n, x)))
    rule6 = ReplacementRule(pattern6, lambda b, c, n, a, x : Subst(Int(x**n*sin(a/b - x/b), x), x, a + b*acos(c*x))/(b*c))
    rubi.add(rule6)

    pattern7 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/x_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n: PositiveIntegerQ(n)))
    rule7 = ReplacementRule(pattern7, lambda b, c, n, a, x : Subst(Int((a + b*x)**n/tan(x), x), x, asin(c*x)))
    rubi.add(rule7)

    pattern8 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/x_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n: PositiveIntegerQ(n)))
    rule8 = ReplacementRule(pattern8, lambda b, c, n, a, x : -Subst(Int((a + b*x)**n/cot(x), x), x, acos(c*x)))
    rubi.add(rule8)

    pattern9 = Pattern(Integral((x_*WC('d', S(1)))**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda m: NonzeroQ(m + S(1))))
    rule9 = ReplacementRule(pattern9, lambda b, c, m, d, n, a, x : -b*c*n*Int((d*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(-1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(d*(m + S(1))) + (d*x)**(m + S(1))*(a + b*asin(c*x))**n/(d*(m + S(1))))
    rubi.add(rule9)

    pattern10 = Pattern(Integral((x_*WC('d', S(1)))**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda m: NonzeroQ(m + S(1))))
    rule10 = ReplacementRule(pattern10, lambda b, c, m, d, n, a, x : b*c*n*Int((d*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(-1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(d*(m + S(1))) + (d*x)**(m + S(1))*(a + b*acos(c*x))**n/(d*(m + S(1))))
    rubi.add(rule10)

    pattern11 = Pattern(Integral(x_**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda m: PositiveIntegerQ(m)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))))
    rule11 = ReplacementRule(pattern11, lambda b, c, m, n, a, x : -b*c*n*Int(x**(m + S(1))*(a + b*asin(c*x))**(n + S(-1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(m + S(1)) + x**(m + S(1))*(a + b*asin(c*x))**n/(m + S(1)))
    rubi.add(rule11)

    pattern12 = Pattern(Integral(x_**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda m: PositiveIntegerQ(m)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))))
    rule12 = ReplacementRule(pattern12, lambda b, c, m, n, a, x : b*c*n*Int(x**(m + S(1))*(a + b*acos(c*x))**(n + S(-1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(m + S(1)) + x**(m + S(1))*(a + b*acos(c*x))**n/(m + S(1)))
    rubi.add(rule12)

    pattern13 = Pattern(Integral(x_**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda m: PositiveIntegerQ(m)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Inequality(S(-2), LessEqual, n, Less, S(-1))))
    rule13 = ReplacementRule(pattern13, lambda b, c, m, n, a, x : -c**(-m + S(-1))*Subst(Int(ExpandTrigReduce((a + b*x)**(n + S(1)), (m - (m + S(1))*sin(x)**S(2))*sin(x)**(m + S(-1)), x), x), x, asin(c*x))/(b*(n + S(1))) + x**m*(a + b*asin(c*x))**(n + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule13)

    pattern14 = Pattern(Integral(x_**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda m: PositiveIntegerQ(m)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Inequality(S(-2), LessEqual, n, Less, S(-1))))
    rule14 = ReplacementRule(pattern14, lambda b, c, m, n, a, x : -c**(-m + S(-1))*Subst(Int(ExpandTrigReduce((a + b*x)**(n + S(1)), (m - (m + S(1))*cos(x)**S(2))*cos(x)**(m + S(-1)), x), x), x, acos(c*x))/(b*(n + S(1))) - x**m*(a + b*acos(c*x))**(n + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule14)

    pattern15 = Pattern(Integral(x_**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda m: PositiveIntegerQ(m)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-2))))
    rule15 = ReplacementRule(pattern15, lambda b, c, m, n, a, x : c*(m + S(1))*Int(x**(m + S(1))*(a + b*asin(c*x))**(n + S(1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(b*(n + S(1))) - m*Int(x**(m + S(-1))*(a + b*asin(c*x))**(n + S(1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(b*c*(n + S(1))) + x**m*(a + b*asin(c*x))**(n + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule15)

    pattern16 = Pattern(Integral(x_**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda m: PositiveIntegerQ(m)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-2))))
    rule16 = ReplacementRule(pattern16, lambda b, c, m, n, a, x : -c*(m + S(1))*Int(x**(m + S(1))*(a + b*acos(c*x))**(n + S(1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(b*(n + S(1))) + m*Int(x**(m + S(-1))*(a + b*acos(c*x))**(n + S(1))/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(b*c*(n + S(1))) - x**m*(a + b*acos(c*x))**(n + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule16)

    pattern17 = Pattern(Integral(x_**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda m: PositiveIntegerQ(m)))
    rule17 = ReplacementRule(pattern17, lambda b, c, m, n, a, x : c**(-m + S(-1))*Subst(Int((a + b*x)**n*sin(x)**m*cos(x), x), x, asin(c*x)))
    rubi.add(rule17)

    pattern18 = Pattern(Integral(x_**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda m: PositiveIntegerQ(m)))
    rule18 = ReplacementRule(pattern18, lambda b, c, m, n, a, x : -c**(-m + S(-1))*Subst(Int((a + b*x)**n*sin(x)*cos(x)**m, x), x, acos(c*x)))
    rubi.add(rule18)

    pattern19 = Pattern(Integral((x_*WC('d', S(1)))**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda n, x: FreeQ(n, x)))
    rule19 = ReplacementRule(pattern19, lambda b, c, m, d, n, a, x : Int((d*x)**m*(a + b*asin(c*x))**n, x))
    rubi.add(rule19)

    pattern20 = Pattern(Integral((x_*WC('d', S(1)))**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda n, x: FreeQ(n, x)))
    rule20 = ReplacementRule(pattern20, lambda b, c, m, d, n, a, x : Int((d*x)**m*(a + b*acos(c*x))**n, x))
    rubi.add(rule20)

    pattern21 = Pattern(Integral(S(1)/(sqrt(d_ + x_**S(2)*WC('e', S(1)))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: PositiveQ(d)))
    rule21 = ReplacementRule(pattern21, lambda b, e, c, d, a, x : log(a + b*asin(c*x))/(b*c*sqrt(d)))
    rubi.add(rule21)

    pattern22 = Pattern(Integral(S(1)/(sqrt(d_ + x_**S(2)*WC('e', S(1)))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: PositiveQ(d)))
    rule22 = ReplacementRule(pattern22, lambda b, e, c, d, a, x : -log(a + b*acos(c*x))/(b*c*sqrt(d)))
    rubi.add(rule22)

    pattern23 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: PositiveQ(d)), CustomConstraint(lambda n: NonzeroQ(n + S(1))))
    rule23 = ReplacementRule(pattern23, lambda b, e, c, d, n, a, x : (a + b*asin(c*x))**(n + S(1))/(b*c*sqrt(d)*(n + S(1))))
    rubi.add(rule23)

    pattern24 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: PositiveQ(d)), CustomConstraint(lambda n: NonzeroQ(n + S(1))))
    rule24 = ReplacementRule(pattern24, lambda b, e, c, d, n, a, x : -(a + b*acos(c*x))**(n + S(1))/(b*c*sqrt(d)*(n + S(1))))
    rubi.add(rule24)

    pattern25 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: Not(PositiveQ(d))))
    rule25 = ReplacementRule(pattern25, lambda b, e, c, d, n, a, x : sqrt(-c**S(2)*x**S(2) + S(1))*Int((a + b*asin(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/sqrt(d + e*x**S(2)))
    rubi.add(rule25)

    pattern26 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: Not(PositiveQ(d))))
    rule26 = ReplacementRule(pattern26, lambda b, e, c, d, n, a, x : sqrt(-c**S(2)*x**S(2) + S(1))*Int((a + b*acos(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/sqrt(d + e*x**S(2)))
    rubi.add(rule26)

    pattern27 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p)), )
    def With27(b, e, c, d, a, p, x):
        u = IntHide((d + e*x**S(2))**p, x)
        return -b*c*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x) + Dist(a + b*asin(c*x), u, x)
    rule27 = ReplacementRule(pattern27, lambda b, e, c, d, a, p, x : With27(b, e, c, d, a, p, x))
    rubi.add(rule27)

    pattern28 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p)), )
    def With28(b, e, c, d, a, p, x):
        u = IntHide((d + e*x**S(2))**p, x)
        return b*c*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x) + Dist(a + b*acos(c*x), u, x)
    rule28 = ReplacementRule(pattern28, lambda b, e, c, d, a, p, x : With28(b, e, c, d, a, p, x))
    rubi.add(rule28)

    pattern29 = Pattern(Integral(sqrt(d_ + x_**S(2)*WC('e', S(1)))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))))
    rule29 = ReplacementRule(pattern29, lambda b, e, c, d, n, a, x : -b*c*n*sqrt(d + e*x**S(2))*Int(x*(a + b*asin(c*x))**(n + S(-1)), x)/(S(2)*sqrt(-c**S(2)*x**S(2) + S(1))) + x*(a + b*asin(c*x))**n*sqrt(d + e*x**S(2))/S(2) + sqrt(d + e*x**S(2))*Int((a + b*asin(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(S(2)*sqrt(-c**S(2)*x**S(2) + S(1))))
    rubi.add(rule29)

    pattern30 = Pattern(Integral(sqrt(d_ + x_**S(2)*WC('e', S(1)))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))))
    rule30 = ReplacementRule(pattern30, lambda b, e, c, d, n, a, x : b*c*n*sqrt(d + e*x**S(2))*Int(x*(a + b*acos(c*x))**(n + S(-1)), x)/(S(2)*sqrt(-c**S(2)*x**S(2) + S(1))) + x*(a + b*acos(c*x))**n*sqrt(d + e*x**S(2))/S(2) + sqrt(d + e*x**S(2))*Int((a + b*acos(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(S(2)*sqrt(-c**S(2)*x**S(2) + S(1))))
    rubi.add(rule30)

    pattern31 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p: RationalQ(n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Greater(p, S(0))))
    rule31 = ReplacementRule(pattern31, lambda b, e, c, d, n, a, p, x : -b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int(x*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(S(2)*p + S(1)) + S(2)*d*p*Int((a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(-1)), x)/(S(2)*p + S(1)) + x*(a + b*asin(c*x))**n*(d + e*x**S(2))**p/(S(2)*p + S(1)))
    rubi.add(rule31)

    pattern32 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p: RationalQ(n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Greater(p, S(0))))
    rule32 = ReplacementRule(pattern32, lambda b, e, c, d, n, a, p, x : b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int(x*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(S(2)*p + S(1)) + S(2)*d*p*Int((a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(-1)), x)/(S(2)*p + S(1)) + x*(a + b*acos(c*x))**n*(d + e*x**S(2))**p/(S(2)*p + S(1)))
    rubi.add(rule32)

    pattern33 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/(d_ + x_**S(2)*WC('e', S(1)))**(S(3)/2), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda d: PositiveQ(d)))
    rule33 = ReplacementRule(pattern33, lambda b, e, c, d, n, a, x : -b*c*n*Int(x*(a + b*asin(c*x))**(n + S(-1))/(d + e*x**S(2)), x)/sqrt(d) + x*(a + b*asin(c*x))**n/(d*sqrt(d + e*x**S(2))))
    rubi.add(rule33)

    pattern34 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/(d_ + x_**S(2)*WC('e', S(1)))**(S(3)/2), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda d: PositiveQ(d)))
    rule34 = ReplacementRule(pattern34, lambda b, e, c, d, n, a, x : b*c*n*Int(x*(a + b*acos(c*x))**(n + S(-1))/(d + e*x**S(2)), x)/sqrt(d) + x*(a + b*acos(c*x))**n/(d*sqrt(d + e*x**S(2))))
    rubi.add(rule34)

    pattern35 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/(d_ + x_**S(2)*WC('e', S(1)))**(S(3)/2), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))))
    rule35 = ReplacementRule(pattern35, lambda b, e, c, d, n, a, x : -b*c*n*sqrt(-c**S(2)*x**S(2) + S(1))*Int(x*(a + b*asin(c*x))**(n + S(-1))/(-c**S(2)*x**S(2) + S(1)), x)/(d*sqrt(d + e*x**S(2))) + x*(a + b*asin(c*x))**n/(d*sqrt(d + e*x**S(2))))
    rubi.add(rule35)

    pattern36 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/(d_ + x_**S(2)*WC('e', S(1)))**(S(3)/2), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))))
    rule36 = ReplacementRule(pattern36, lambda b, e, c, d, n, a, x : b*c*n*sqrt(-c**S(2)*x**S(2) + S(1))*Int(x*(a + b*acos(c*x))**(n + S(-1))/(-c**S(2)*x**S(2) + S(1)), x)/(d*sqrt(d + e*x**S(2))) + x*(a + b*acos(c*x))**n/(d*sqrt(d + e*x**S(2))))
    rubi.add(rule36)

    pattern37 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p: RationalQ(n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Less(p, S(-1))), CustomConstraint(lambda p: Unequal(p, S(-3)/2)))
    rule37 = ReplacementRule(pattern37, lambda b, e, c, d, n, a, p, x : b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int(x*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(S(2)*(p + S(1))) - x*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1))/(S(2)*d*(p + S(1))) + (S(2)*p + S(3))*Int((a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1)), x)/(S(2)*d*(p + S(1))))
    rubi.add(rule37)

    pattern38 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p: RationalQ(n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Less(p, S(-1))), CustomConstraint(lambda p: Unequal(p, S(-3)/2)))
    rule38 = ReplacementRule(pattern38, lambda b, e, c, d, n, a, p, x : -b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int(x*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(S(2)*(p + S(1))) - x*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1))/(S(2)*d*(p + S(1))) + (S(2)*p + S(3))*Int((a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1)), x)/(S(2)*d*(p + S(1))))
    rubi.add(rule38)

    pattern39 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: PositiveIntegerQ(n)))
    rule39 = ReplacementRule(pattern39, lambda b, e, c, d, n, a, x : Subst(Int((a + b*x)**n*sec(x), x), x, asin(c*x))/(c*d))
    rubi.add(rule39)

    pattern40 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: PositiveIntegerQ(n)))
    rule40 = ReplacementRule(pattern40, lambda b, e, c, d, n, a, x : -Subst(Int((a + b*x)**n*csc(x), x), x, acos(c*x))/(c*d))
    rubi.add(rule40)

    pattern41 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))))
    rule41 = ReplacementRule(pattern41, lambda b, e, c, d, n, a, p, x : c*d**IntPart(p)*(d + e*x**S(2))**FracPart(p)*(S(2)*p + S(1))*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int(x*(a + b*asin(c*x))**(n + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(b*(n + S(1))) + (a + b*asin(c*x))**(n + S(1))*(d + e*x**S(2))**p*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule41)

    pattern42 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))))
    rule42 = ReplacementRule(pattern42, lambda b, e, c, d, n, a, p, x : -c*d**IntPart(p)*(d + e*x**S(2))**FracPart(p)*(S(2)*p + S(1))*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int(x*(a + b*acos(c*x))**(n + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(b*(n + S(1))) - (a + b*acos(c*x))**(n + S(1))*(d + e*x**S(2))**p*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule42)

    pattern43 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(S(2)*p)), CustomConstraint(lambda p, d: IntegerQ(p) | PositiveQ(d)))
    rule43 = ReplacementRule(pattern43, lambda b, e, c, d, n, a, p, x : d**p*Subst(Int((a + b*x)**n*cos(x)**(S(2)*p + S(1)), x), x, asin(c*x))/c)
    rubi.add(rule43)

    pattern44 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(S(2)*p)), CustomConstraint(lambda p, d: IntegerQ(p) | PositiveQ(d)))
    rule44 = ReplacementRule(pattern44, lambda b, e, c, d, n, a, p, x : -d**p*Subst(Int((a + b*x)**n*sin(x)**(S(2)*p + S(1)), x), x, acos(c*x))/c)
    rubi.add(rule44)

    pattern45 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(S(2)*p)), CustomConstraint(lambda p, d: Not(IntegerQ(p) | PositiveQ(d))))
    rule45 = ReplacementRule(pattern45, lambda b, e, c, d, n, a, p, x : d**(p + S(-1)/2)*sqrt(d + e*x**S(2))*Int((a + b*asin(c*x))**n*(-c**S(2)*x**S(2) + S(1))**p, x)/sqrt(-c**S(2)*x**S(2) + S(1)))
    rubi.add(rule45)

    pattern46 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(S(2)*p)), CustomConstraint(lambda p, d: Not(IntegerQ(p) | PositiveQ(d))))
    rule46 = ReplacementRule(pattern46, lambda b, e, c, d, n, a, p, x : d**(p + S(-1)/2)*sqrt(d + e*x**S(2))*Int((a + b*acos(c*x))**n*(-c**S(2)*x**S(2) + S(1))**p, x)/sqrt(-c**S(2)*x**S(2) + S(1)))
    rubi.add(rule46)

    pattern47 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: NonzeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p) | NegativeIntegerQ(p + S(1)/2)), )
    def With47(b, e, c, d, a, p, x):
        u = IntHide((d + e*x**S(2))**p, x)
        return -b*c*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x) + Dist(a + b*asin(c*x), u, x)
    rule47 = ReplacementRule(pattern47, lambda b, e, c, d, a, p, x : With47(b, e, c, d, a, p, x))
    rubi.add(rule47)

    pattern48 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: NonzeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p) | NegativeIntegerQ(p + S(1)/2)), )
    def With48(b, e, c, d, a, p, x):
        u = IntHide((d + e*x**S(2))**p, x)
        return b*c*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x) + Dist(a + b*acos(c*x), u, x)
    rule48 = ReplacementRule(pattern48, lambda b, e, c, d, a, p, x : With48(b, e, c, d, a, p, x))
    rubi.add(rule48)

    pattern49 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: NonzeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: IntegerQ(p)), CustomConstraint(lambda n, p: PositiveIntegerQ(n) | Greater(p, S(0))))
    rule49 = ReplacementRule(pattern49, lambda b, e, c, d, n, a, p, x : Int(ExpandIntegrand((a + b*asin(c*x))**n, (d + e*x**S(2))**p, x), x))
    rubi.add(rule49)

    pattern50 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda e, c, d: NonzeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: IntegerQ(p)), CustomConstraint(lambda n, p: PositiveIntegerQ(n) | Greater(p, S(0))))
    rule50 = ReplacementRule(pattern50, lambda b, e, c, d, n, a, p, x : Int(ExpandIntegrand((a + b*acos(c*x))**n, (d + e*x**S(2))**p, x), x))
    rubi.add(rule50)

    pattern51 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda p, x: FreeQ(p, x)))
    rule51 = ReplacementRule(pattern51, lambda b, e, c, d, n, a, p, x : Int((a + b*asin(c*x))**n*(d + e*x**S(2))**p, x))
    rubi.add(rule51)

    pattern52 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda p, x: FreeQ(p, x)))
    rule52 = ReplacementRule(pattern52, lambda b, e, c, d, n, a, p, x : Int((a + b*acos(c*x))**n*(d + e*x**S(2))**p, x))
    rubi.add(rule52)

    pattern53 = Pattern(Integral((d_ + x_*WC('e', S(1)))**p_*(f_ + x_*WC('g', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda g, x: FreeQ(g, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda f, e, g, d: ZeroQ(d*g + e*f)), CustomConstraint(lambda f, g, c: ZeroQ(c**S(2)*f**S(2) - g**S(2))), CustomConstraint(lambda p: Not(IntegerQ(p))))
    rule53 = ReplacementRule(pattern53, lambda b, e, c, d, n, g, a, f, p, x : (d + e*x)**FracPart(p)*(f + g*x)**FracPart(p)*(d*f + e*g*x**S(2))**(-FracPart(p))*Int((a + b*asin(c*x))**n*(d*f + e*g*x**S(2))**p, x))
    rubi.add(rule53)

    pattern54 = Pattern(Integral((d_ + x_*WC('e', S(1)))**p_*(f_ + x_*WC('g', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda g, x: FreeQ(g, x)), CustomConstraint(lambda n, x: FreeQ(n, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda f, e, g, d: ZeroQ(d*g + e*f)), CustomConstraint(lambda f, g, c: ZeroQ(c**S(2)*f**S(2) - g**S(2))), CustomConstraint(lambda p: Not(IntegerQ(p))))
    rule54 = ReplacementRule(pattern54, lambda b, e, c, d, n, g, a, f, p, x : (d + e*x)**FracPart(p)*(f + g*x)**FracPart(p)*(d*f + e*g*x**S(2))**(-FracPart(p))*Int((a + b*acos(c*x))**n*(d*f + e*g*x**S(2))**p, x))
    rubi.add(rule54)

    pattern55 = Pattern(Integral(x_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: PositiveIntegerQ(n)))
    rule55 = ReplacementRule(pattern55, lambda b, e, c, d, n, a, x : -Subst(Int((a + b*x)**n*tan(x), x), x, asin(c*x))/e)
    rubi.add(rule55)

    pattern56 = Pattern(Integral(x_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: PositiveIntegerQ(n)))
    rule56 = ReplacementRule(pattern56, lambda b, e, c, d, n, a, x : Subst(Int((a + b*x)**n*cot(x), x), x, acos(c*x))/e)
    rubi.add(rule56)

    pattern57 = Pattern(Integral(x_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: NonzeroQ(p + S(1))))
    rule57 = ReplacementRule(pattern57, lambda b, e, c, d, n, a, p, x : b*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(S(2)*c*(p + S(1))) + (a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1))/(S(2)*e*(p + S(1))))
    rubi.add(rule57)

    pattern58 = Pattern(Integral(x_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: NonzeroQ(p + S(1))))
    rule58 = ReplacementRule(pattern58, lambda b, e, c, d, n, a, p, x : -b*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(S(2)*c*(p + S(1))) + (a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1))/(S(2)*e*(p + S(1))))
    rubi.add(rule58)

    pattern59 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/(x_*(d_ + x_**S(2)*WC('e', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: PositiveIntegerQ(n)))
    rule59 = ReplacementRule(pattern59, lambda b, e, c, d, n, a, x : Subst(Int((a + b*x)**n/(sin(x)*cos(x)), x), x, asin(c*x))/d)
    rubi.add(rule59)

    pattern60 = Pattern(Integral((WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/(x_*(d_ + x_**S(2)*WC('e', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: PositiveIntegerQ(n)))
    rule60 = ReplacementRule(pattern60, lambda b, e, c, d, n, a, x : -Subst(Int((a + b*x)**n/(sin(x)*cos(x)), x), x, acos(c*x))/d)
    rubi.add(rule60)

    pattern61 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p, m: ZeroQ(m + S(2)*p + S(3))), CustomConstraint(lambda m: NonzeroQ(m + S(1))))
    rule61 = ReplacementRule(pattern61, lambda b, e, c, m, d, n, a, f, p, x : -b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(f*(m + S(1))) + (f*x)**(m + S(1))*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1))/(d*f*(m + S(1))))
    rubi.add(rule61)

    pattern62 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p, m: ZeroQ(m + S(2)*p + S(3))), CustomConstraint(lambda m: NonzeroQ(m + S(1))))
    rule62 = ReplacementRule(pattern62, lambda b, e, c, m, d, n, a, f, p, x : b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(f*(m + S(1))) + (f*x)**(m + S(1))*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1))/(d*f*(m + S(1))))
    rubi.add(rule62)

    pattern63 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))/x_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p)))
    rule63 = ReplacementRule(pattern63, lambda b, e, c, d, a, p, x : -b*c*d**p*Int((-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(S(2)*p) + d*Int((a + b*asin(c*x))*(d + e*x**S(2))**(p + S(-1))/x, x) + (a + b*asin(c*x))*(d + e*x**S(2))**p/(S(2)*p))
    rubi.add(rule63)

    pattern64 = Pattern(Integral((d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))/x_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p)))
    rule64 = ReplacementRule(pattern64, lambda b, e, c, d, a, p, x : b*c*d**p*Int((-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(S(2)*p) + d*Int((a + b*acos(c*x))*(d + e*x**S(2))**(p + S(-1))/x, x) + (a + b*acos(c*x))*(d + e*x**S(2))**p/(S(2)*p))
    rubi.add(rule64)

    pattern65 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p)), CustomConstraint(lambda m: NegativeIntegerQ(m/S(2) + S(1)/2)))
    rule65 = ReplacementRule(pattern65, lambda b, e, c, m, d, a, f, p, x : -b*c*d**p*Int((f*x)**(m + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(f*(m + S(1))) - S(2)*e*p*Int((f*x)**(m + S(2))*(a + b*asin(c*x))*(d + e*x**S(2))**(p + S(-1)), x)/(f**S(2)*(m + S(1))) + (f*x)**(m + S(1))*(a + b*asin(c*x))*(d + e*x**S(2))**p/(f*(m + S(1))))
    rubi.add(rule65)

    pattern66 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p)), CustomConstraint(lambda m: NegativeIntegerQ(m/S(2) + S(1)/2)))
    rule66 = ReplacementRule(pattern66, lambda b, e, c, m, d, a, f, p, x : b*c*d**p*Int((f*x)**(m + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(f*(m + S(1))) - S(2)*e*p*Int((f*x)**(m + S(2))*(a + b*acos(c*x))*(d + e*x**S(2))**(p + S(-1)), x)/(f**S(2)*(m + S(1))) + (f*x)**(m + S(1))*(a + b*acos(c*x))*(d + e*x**S(2))**p/(f*(m + S(1))))
    rubi.add(rule66)

    pattern67 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p)), )
    def With67(b, e, c, m, d, a, f, p, x):
        u = IntHide((f*x)**m*(d + e*x**S(2))**p, x)
        return -b*c*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x) + Dist(a + b*asin(c*x), u, x)
    rule67 = ReplacementRule(pattern67, lambda b, e, c, m, d, a, f, p, x : With67(b, e, c, m, d, a, f, p, x))
    rubi.add(rule67)

    pattern68 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p)), )
    def With68(b, e, c, m, d, a, f, p, x):
        u = IntHide((f*x)**m*(d + e*x**S(2))**p, x)
        return b*c*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x) + Dist(a + b*acos(c*x), u, x)
    rule68 = ReplacementRule(pattern68, lambda b, e, c, m, d, a, f, p, x : With68(b, e, c, m, d, a, f, p, x))
    rubi.add(rule68)

    pattern69 = Pattern(Integral(x_**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: IntegerQ(p + S(-1)/2)), CustomConstraint(lambda p, m: PositiveIntegerQ(m/S(2) + S(1)/2) | NegativeIntegerQ(m/S(2) + p + S(3)/2)), CustomConstraint(lambda p: Unequal(p, S(-1)/2)), CustomConstraint(lambda d: PositiveQ(d)), )
    def With69(b, e, c, m, d, a, p, x):
        u = IntHide(x**m*(-c**S(2)*x**S(2) + S(1))**p, x)
        return -b*c*d**p*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x) + Dist(d**p*(a + b*asin(c*x)), u, x)
    rule69 = ReplacementRule(pattern69, lambda b, e, c, m, d, a, p, x : With69(b, e, c, m, d, a, p, x))
    rubi.add(rule69)

    pattern70 = Pattern(Integral(x_**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: IntegerQ(p + S(-1)/2)), CustomConstraint(lambda p, m: PositiveIntegerQ(m/S(2) + S(1)/2) | NegativeIntegerQ(m/S(2) + p + S(3)/2)), CustomConstraint(lambda p: Unequal(p, S(-1)/2)), CustomConstraint(lambda d: PositiveQ(d)), )
    def With70(b, e, c, m, d, a, p, x):
        u = IntHide(x**m*(-c**S(2)*x**S(2) + S(1))**p, x)
        return b*c*d**p*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x) + Dist(d**p*(a + b*acos(c*x)), u, x)
    rule70 = ReplacementRule(pattern70, lambda b, e, c, m, d, a, p, x : With70(b, e, c, m, d, a, p, x))
    rubi.add(rule70)

    pattern71 = Pattern(Integral(x_**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p + S(1)/2)), CustomConstraint(lambda p, m: PositiveIntegerQ(m/S(2) + S(1)/2) | NegativeIntegerQ(m/S(2) + p + S(3)/2)), )
    def With71(b, e, c, m, d, a, p, x):
        u = IntHide(x**m*(-c**S(2)*x**S(2) + S(1))**p, x)
        return -b*c*d**(p + S(-1)/2)*sqrt(d + e*x**S(2))*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x)/sqrt(-c**S(2)*x**S(2) + S(1)) + (a + b*asin(c*x))*Int(x**m*(d + e*x**S(2))**p, x)
    rule71 = ReplacementRule(pattern71, lambda b, e, c, m, d, a, p, x : With71(b, e, c, m, d, a, p, x))
    rubi.add(rule71)

    pattern72 = Pattern(Integral(x_**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1)))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda p: PositiveIntegerQ(p + S(1)/2)), CustomConstraint(lambda p, m: PositiveIntegerQ(m/S(2) + S(1)/2) | NegativeIntegerQ(m/S(2) + p + S(3)/2)), )
    def With72(b, e, c, m, d, a, p, x):
        u = IntHide(x**m*(-c**S(2)*x**S(2) + S(1))**p, x)
        return b*c*d**(p + S(-1)/2)*sqrt(d + e*x**S(2))*Int(SimplifyIntegrand(u/sqrt(-c**S(2)*x**S(2) + S(1)), x), x)/sqrt(-c**S(2)*x**S(2) + S(1)) + (a + b*acos(c*x))*Int(x**m*(d + e*x**S(2))**p, x)
    rule72 = ReplacementRule(pattern72, lambda b, e, c, m, d, a, p, x : With72(b, e, c, m, d, a, p, x))
    rubi.add(rule72)

    pattern73 = Pattern(Integral((x_*WC('f', S(1)))**m_*sqrt(d_ + x_**S(2)*WC('e', S(1)))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, m: RationalQ(m, n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Less(m, S(-1))))
    rule73 = ReplacementRule(pattern73, lambda b, e, c, m, d, n, a, f, x : -b*c*n*sqrt(d + e*x**S(2))*Int((f*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(-1)), x)/(f*(m + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))) + c**S(2)*sqrt(d + e*x**S(2))*Int((f*x)**(m + S(2))*(a + b*asin(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(f**S(2)*(m + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))) + (f*x)**(m + S(1))*(a + b*asin(c*x))**n*sqrt(d + e*x**S(2))/(f*(m + S(1))))
    rubi.add(rule73)

    pattern74 = Pattern(Integral((x_*WC('f', S(1)))**m_*sqrt(d_ + x_**S(2)*WC('e', S(1)))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, m: RationalQ(m, n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Less(m, S(-1))))
    rule74 = ReplacementRule(pattern74, lambda b, e, c, m, d, n, a, f, x : b*c*n*sqrt(d + e*x**S(2))*Int((f*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(-1)), x)/(f*(m + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))) + c**S(2)*sqrt(d + e*x**S(2))*Int((f*x)**(m + S(2))*(a + b*acos(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/(f**S(2)*(m + S(1))*sqrt(-c**S(2)*x**S(2) + S(1))) + (f*x)**(m + S(1))*(a + b*acos(c*x))**n*sqrt(d + e*x**S(2))/(f*(m + S(1))))
    rubi.add(rule74)

    pattern75 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p, m: RationalQ(m, n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Greater(p, S(0))), CustomConstraint(lambda m: Less(m, S(-1))))
    rule75 = ReplacementRule(pattern75, lambda b, e, c, m, d, n, a, f, p, x : -b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(f*(m + S(1))) - S(2)*e*p*Int((f*x)**(m + S(2))*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(-1)), x)/(f**S(2)*(m + S(1))) + (f*x)**(m + S(1))*(a + b*asin(c*x))**n*(d + e*x**S(2))**p/(f*(m + S(1))))
    rubi.add(rule75)

    pattern76 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p, m: RationalQ(m, n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Greater(p, S(0))), CustomConstraint(lambda m: Less(m, S(-1))))
    rule76 = ReplacementRule(pattern76, lambda b, e, c, m, d, n, a, f, p, x : b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(f*(m + S(1))) - S(2)*e*p*Int((f*x)**(m + S(2))*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(-1)), x)/(f**S(2)*(m + S(1))) + (f*x)**(m + S(1))*(a + b*acos(c*x))**n*(d + e*x**S(2))**p/(f*(m + S(1))))
    rubi.add(rule76)

    pattern77 = Pattern(Integral((x_*WC('f', S(1)))**m_*sqrt(d_ + x_**S(2)*WC('e', S(1)))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Not(RationalQ(m) & Less(m, S(-1)))), CustomConstraint(lambda n, m: RationalQ(m) | ZeroQ(n + S(-1))))
    rule77 = ReplacementRule(pattern77, lambda b, e, c, m, d, n, a, f, x : -b*c*n*sqrt(d + e*x**S(2))*Int((f*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(-1)), x)/(f*(m + S(2))*sqrt(-c**S(2)*x**S(2) + S(1))) + sqrt(d + e*x**S(2))*Int((f*x)**m*(a + b*asin(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/((m + S(2))*sqrt(-c**S(2)*x**S(2) + S(1))) + (f*x)**(m + S(1))*(a + b*asin(c*x))**n*sqrt(d + e*x**S(2))/(f*(m + S(2))))
    rubi.add(rule77)

    pattern78 = Pattern(Integral((x_*WC('f', S(1)))**m_*sqrt(d_ + x_**S(2)*WC('e', S(1)))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Not(RationalQ(m) & Less(m, S(-1)))), CustomConstraint(lambda n, m: RationalQ(m) | ZeroQ(n + S(-1))))
    rule78 = ReplacementRule(pattern78, lambda b, e, c, m, d, n, a, f, x : b*c*n*sqrt(d + e*x**S(2))*Int((f*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(-1)), x)/(f*(m + S(2))*sqrt(-c**S(2)*x**S(2) + S(1))) + sqrt(d + e*x**S(2))*Int((f*x)**m*(a + b*acos(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/((m + S(2))*sqrt(-c**S(2)*x**S(2) + S(1))) + (f*x)**(m + S(1))*(a + b*acos(c*x))**n*sqrt(d + e*x**S(2))/(f*(m + S(2))))
    rubi.add(rule78)

    pattern79 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p: RationalQ(n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Greater(p, S(0))), CustomConstraint(lambda m: Not(RationalQ(m) & Less(m, S(-1)))), CustomConstraint(lambda n, m: RationalQ(m) | ZeroQ(n + S(-1))))
    rule79 = ReplacementRule(pattern79, lambda b, e, c, m, d, n, a, f, p, x : -b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(f*(m + S(2)*p + S(1))) + S(2)*d*p*Int((f*x)**m*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(-1)), x)/(m + S(2)*p + S(1)) + (f*x)**(m + S(1))*(a + b*asin(c*x))**n*(d + e*x**S(2))**p/(f*(m + S(2)*p + S(1))))
    rubi.add(rule79)

    pattern80 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p: RationalQ(n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Greater(p, S(0))), CustomConstraint(lambda m: Not(RationalQ(m) & Less(m, S(-1)))), CustomConstraint(lambda n, m: RationalQ(m) | ZeroQ(n + S(-1))))
    rule80 = ReplacementRule(pattern80, lambda b, e, c, m, d, n, a, f, p, x : b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(f*(m + S(2)*p + S(1))) + S(2)*d*p*Int((f*x)**m*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(-1)), x)/(m + S(2)*p + S(1)) + (f*x)**(m + S(1))*(a + b*acos(c*x))**n*(d + e*x**S(2))**p/(f*(m + S(2)*p + S(1))))
    rubi.add(rule80)

    pattern81 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, m: RationalQ(m, n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Less(m, S(-1))), CustomConstraint(lambda m: IntegerQ(m)))
    rule81 = ReplacementRule(pattern81, lambda b, e, c, m, d, n, a, f, p, x : -b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(f*(m + S(1))) + c**S(2)*(m + S(2)*p + S(3))*Int((f*x)**(m + S(2))*(a + b*asin(c*x))**n*(d + e*x**S(2))**p, x)/(f**S(2)*(m + S(1))) + (f*x)**(m + S(1))*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1))/(d*f*(m + S(1))))
    rubi.add(rule81)

    pattern82 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, m: RationalQ(m, n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Less(m, S(-1))), CustomConstraint(lambda m: IntegerQ(m)))
    rule82 = ReplacementRule(pattern82, lambda b, e, c, m, d, n, a, f, p, x : b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(f*(m + S(1))) + c**S(2)*(m + S(2)*p + S(3))*Int((f*x)**(m + S(2))*(a + b*acos(c*x))**n*(d + e*x**S(2))**p, x)/(f**S(2)*(m + S(1))) + (f*x)**(m + S(1))*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1))/(d*f*(m + S(1))))
    rubi.add(rule82)

    pattern83 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p, m: RationalQ(m, n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Less(p, S(-1))), CustomConstraint(lambda m: Greater(m, S(1))))
    rule83 = ReplacementRule(pattern83, lambda b, e, c, m, d, n, a, f, p, x : b*d**IntPart(p)*f*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(-1))*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(S(2)*c*(p + S(1))) - f**S(2)*(m + S(-1))*Int((f*x)**(m + S(-2))*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1)), x)/(S(2)*e*(p + S(1))) + f*(f*x)**(m + S(-1))*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1))/(S(2)*e*(p + S(1))))
    rubi.add(rule83)

    pattern84 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p, m: RationalQ(m, n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Less(p, S(-1))), CustomConstraint(lambda m: Greater(m, S(1))))
    rule84 = ReplacementRule(pattern84, lambda b, e, c, m, d, n, a, f, p, x : -b*d**IntPart(p)*f*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(-1))*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(S(2)*c*(p + S(1))) - f**S(2)*(m + S(-1))*Int((f*x)**(m + S(-2))*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1)), x)/(S(2)*e*(p + S(1))) + f*(f*x)**(m + S(-1))*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1))/(S(2)*e*(p + S(1))))
    rubi.add(rule84)

    pattern85 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p: RationalQ(n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Less(p, S(-1))), CustomConstraint(lambda m: Not(RationalQ(m) & Greater(m, S(1)))), CustomConstraint(lambda n, p, m: IntegerQ(m) | IntegerQ(p) | Equal(n, S(1))))
    rule85 = ReplacementRule(pattern85, lambda b, e, c, m, d, n, a, f, p, x : b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(S(2)*f*(p + S(1))) + (m + S(2)*p + S(3))*Int((f*x)**m*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1)), x)/(S(2)*d*(p + S(1))) - (f*x)**(m + S(1))*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1))/(S(2)*d*f*(p + S(1))))
    rubi.add(rule85)

    pattern86 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, p: RationalQ(n, p)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda p: Less(p, S(-1))), CustomConstraint(lambda m: Not(RationalQ(m) & Greater(m, S(1)))), CustomConstraint(lambda n, p, m: IntegerQ(m) | IntegerQ(p) | Equal(n, S(1))))
    rule86 = ReplacementRule(pattern86, lambda b, e, c, m, d, n, a, f, p, x : -b*c*d**IntPart(p)*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(S(2)*f*(p + S(1))) + (m + S(2)*p + S(3))*Int((f*x)**m*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1)), x)/(S(2)*d*(p + S(1))) - (f*x)**(m + S(1))*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1))/(S(2)*d*f*(p + S(1))))
    rubi.add(rule86)

    pattern87 = Pattern(Integral((x_*WC('f', S(1)))**m_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, m: RationalQ(m, n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Greater(m, S(1))), CustomConstraint(lambda m: IntegerQ(m)))
    rule87 = ReplacementRule(pattern87, lambda b, e, c, m, d, n, a, f, x : b*f*n*sqrt(-c**S(2)*x**S(2) + S(1))*Int((f*x)**(m + S(-1))*(a + b*asin(c*x))**(n + S(-1)), x)/(c*m*sqrt(d + e*x**S(2))) + f*(f*x)**(m + S(-1))*(a + b*asin(c*x))**n*sqrt(d + e*x**S(2))/(e*m) + f**S(2)*(m + S(-1))*Int((f*x)**(m + S(-2))*(a + b*asin(c*x))**n/sqrt(d + e*x**S(2)), x)/(c**S(2)*m))
    rubi.add(rule87)

    pattern88 = Pattern(Integral((x_*WC('f', S(1)))**m_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, m: RationalQ(m, n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Greater(m, S(1))), CustomConstraint(lambda m: IntegerQ(m)))
    rule88 = ReplacementRule(pattern88, lambda b, e, c, m, d, n, a, f, x : -b*f*n*sqrt(-c**S(2)*x**S(2) + S(1))*Int((f*x)**(m + S(-1))*(a + b*acos(c*x))**(n + S(-1)), x)/(c*m*sqrt(d + e*x**S(2))) + f*(f*x)**(m + S(-1))*(a + b*acos(c*x))**n*sqrt(d + e*x**S(2))/(e*m) + f**S(2)*(m + S(-1))*Int((f*x)**(m + S(-2))*(a + b*acos(c*x))**n/sqrt(d + e*x**S(2)), x)/(c**S(2)*m))
    rubi.add(rule88)

    pattern89 = Pattern(Integral(x_**m_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: PositiveQ(d)), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda m: IntegerQ(m)))
    rule89 = ReplacementRule(pattern89, lambda b, e, c, m, d, n, a, x : c**(-m + S(-1))*Subst(Int((a + b*x)**n*sin(x)**m, x), x, asin(c*x))/sqrt(d))
    rubi.add(rule89)

    pattern90 = Pattern(Integral(x_**m_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: PositiveQ(d)), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda m: IntegerQ(m)))
    rule90 = ReplacementRule(pattern90, lambda b, e, c, m, d, n, a, x : -c**(-m + S(-1))*Subst(Int((a + b*x)**n*cos(x)**m, x), x, acos(c*x))/sqrt(d))
    rubi.add(rule90)

    pattern91 = Pattern(Integral((x_*WC('f', S(1)))**m_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: PositiveQ(d)), CustomConstraint(lambda m: Not(IntegerQ(m))))
    rule91 = ReplacementRule(pattern91, lambda b, e, c, m, d, a, f, x : -b*c*(f*x)**(m + S(2))*HypergeometricPFQ(List(S(1), m/S(2) + S(1), m/S(2) + S(1)), List(m/S(2) + S(3)/2, m/S(2) + S(2)), c**S(2)*x**S(2))/(sqrt(d)*f**S(2)*(m + S(1))*(m + S(2))) + (f*x)**(m + S(1))*(a + b*asin(c*x))*Hypergeometric2F1(S(1)/2, m/S(2) + S(1)/2, m/S(2) + S(3)/2, c**S(2)*x**S(2))/(sqrt(d)*f*(m + S(1))))
    rubi.add(rule91)

    pattern92 = Pattern(Integral((x_*WC('f', S(1)))**m_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda d: PositiveQ(d)), CustomConstraint(lambda m: Not(IntegerQ(m))))
    rule92 = ReplacementRule(pattern92, lambda b, e, c, m, d, a, f, x : b*c*(f*x)**(m + S(2))*HypergeometricPFQ(List(S(1), m/S(2) + S(1), m/S(2) + S(1)), List(m/S(2) + S(3)/2, m/S(2) + S(2)), c**S(2)*x**S(2))/(sqrt(d)*f**S(2)*(m + S(1))*(m + S(2))) + (f*x)**(m + S(1))*(a + b*acos(c*x))*Hypergeometric2F1(S(1)/2, m/S(2) + S(1)/2, m/S(2) + S(3)/2, c**S(2)*x**S(2))/(sqrt(d)*f*(m + S(1))))
    rubi.add(rule92)

    pattern93 = Pattern(Integral((x_*WC('f', S(1)))**m_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda d: Not(PositiveQ(d))), CustomConstraint(lambda n, m: IntegerQ(m) | Equal(n, S(1))))
    rule93 = ReplacementRule(pattern93, lambda b, e, c, m, d, n, a, f, x : sqrt(-c**S(2)*x**S(2) + S(1))*Int((f*x)**m*(a + b*asin(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/sqrt(d + e*x**S(2)))
    rubi.add(rule93)

    pattern94 = Pattern(Integral((x_*WC('f', S(1)))**m_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1))/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda d: Not(PositiveQ(d))), CustomConstraint(lambda n, m: IntegerQ(m) | Equal(n, S(1))))
    rule94 = ReplacementRule(pattern94, lambda b, e, c, m, d, n, a, f, x : sqrt(-c**S(2)*x**S(2) + S(1))*Int((f*x)**m*(a + b*acos(c*x))**n/sqrt(-c**S(2)*x**S(2) + S(1)), x)/sqrt(d + e*x**S(2)))
    rubi.add(rule94)

    pattern95 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, m: RationalQ(m, n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Greater(m, S(1))), CustomConstraint(lambda p, m: NonzeroQ(m + S(2)*p + S(1))), CustomConstraint(lambda m: IntegerQ(m)))
    rule95 = ReplacementRule(pattern95, lambda b, e, c, m, d, n, a, f, p, x : b*d**IntPart(p)*f*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(-1))*(a + b*asin(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(c*(m + S(2)*p + S(1))) + f*(f*x)**(m + S(-1))*(a + b*asin(c*x))**n*(d + e*x**S(2))**(p + S(1))/(e*(m + S(2)*p + S(1))) + f**S(2)*(m + S(-1))*Int((f*x)**(m + S(-2))*(a + b*asin(c*x))**n*(d + e*x**S(2))**p, x)/(c**S(2)*(m + S(2)*p + S(1))))
    rubi.add(rule95)

    pattern96 = Pattern(Integral((x_*WC('f', S(1)))**m_*(d_ + x_**S(2)*WC('e', S(1)))**p_*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**WC('n', S(1)), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n, m: RationalQ(m, n)), CustomConstraint(lambda n: Greater(n, S(0))), CustomConstraint(lambda m: Greater(m, S(1))), CustomConstraint(lambda p, m: NonzeroQ(m + S(2)*p + S(1))), CustomConstraint(lambda m: IntegerQ(m)))
    rule96 = ReplacementRule(pattern96, lambda b, e, c, m, d, n, a, f, p, x : -b*d**IntPart(p)*f*n*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(-1))*(a + b*acos(c*x))**(n + S(-1))*(-c**S(2)*x**S(2) + S(1))**(p + S(1)/2), x)/(c*(m + S(2)*p + S(1))) + f*(f*x)**(m + S(-1))*(a + b*acos(c*x))**n*(d + e*x**S(2))**(p + S(1))/(e*(m + S(2)*p + S(1))) + f**S(2)*(m + S(-1))*Int((f*x)**(m + S(-2))*(a + b*acos(c*x))**n*(d + e*x**S(2))**p, x)/(c**S(2)*(m + S(2)*p + S(1))))
    rubi.add(rule96)

    pattern97 = Pattern(Integral((x_*WC('f', S(1)))**WC('m', S(1))*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))), CustomConstraint(lambda p, m: ZeroQ(m + S(2)*p + S(1))))
    rule97 = ReplacementRule(pattern97, lambda b, e, c, m, d, n, a, f, p, x : -d**IntPart(p)*f*m*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(-1))*(a + b*asin(c*x))**(n + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(b*c*(n + S(1))) + (f*x)**m*(a + b*asin(c*x))**(n + S(1))*(d + e*x**S(2))**p*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule97)

    pattern98 = Pattern(Integral((x_*WC('f', S(1)))**WC('m', S(1))*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda p, x: FreeQ(p, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))), CustomConstraint(lambda p, m: ZeroQ(m + S(2)*p + S(1))))
    rule98 = ReplacementRule(pattern98, lambda b, e, c, m, d, n, a, f, p, x : d**IntPart(p)*f*m*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(-1))*(a + b*acos(c*x))**(n + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(b*c*(n + S(1))) - (f*x)**m*(a + b*acos(c*x))**(n + S(1))*(d + e*x**S(2))**p*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule98)

    pattern99 = Pattern(Integral((x_*WC('f', S(1)))**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))), CustomConstraint(lambda d: PositiveQ(d)))
    rule99 = ReplacementRule(pattern99, lambda b, e, c, m, d, n, a, f, x : -f*m*Int((f*x)**(m + S(-1))*(a + b*asin(c*x))**(n + S(1)), x)/(b*c*sqrt(d)*(n + S(1))) + (f*x)**m*(a + b*asin(c*x))**(n + S(1))/(b*c*sqrt(d)*(n + S(1))))
    rubi.add(rule99)

    pattern100 = Pattern(Integral((x_*WC('f', S(1)))**WC('m', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_/sqrt(d_ + x_**S(2)*WC('e', S(1))), x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda m, x: FreeQ(m, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))), CustomConstraint(lambda d: PositiveQ(d)))
    rule100 = ReplacementRule(pattern100, lambda b, e, c, m, d, n, a, f, x : f*m*Int((f*x)**(m + S(-1))*(a + b*acos(c*x))**(n + S(1)), x)/(b*c*sqrt(d)*(n + S(1))) - (f*x)**m*(a + b*acos(c*x))**(n + S(1))/(b*c*sqrt(d)*(n + S(1))))
    rubi.add(rule100)

    pattern101 = Pattern(Integral((x_*WC('f', S(1)))**WC('m', S(1))*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*asin(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))), CustomConstraint(lambda m: IntegerQ(m)), CustomConstraint(lambda m: Greater(m, S(-3))), CustomConstraint(lambda p: PositiveIntegerQ(S(2)*p)))
    rule101 = ReplacementRule(pattern101, lambda b, e, c, m, d, n, a, f, p, x : c*d**IntPart(p)*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*(m + S(2)*p + S(1))*Int((f*x)**(m + S(1))*(a + b*asin(c*x))**(n + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(b*f*(n + S(1))) - d**IntPart(p)*f*m*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(-1))*(a + b*asin(c*x))**(n + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(b*c*(n + S(1))) + (f*x)**m*(a + b*asin(c*x))**(n + S(1))*(d + e*x**S(2))**p*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule101)

    pattern102 = Pattern(Integral((x_*WC('f', S(1)))**WC('m', S(1))*(d_ + x_**S(2)*WC('e', S(1)))**WC('p', S(1))*(WC('a', S(0)) + WC('b', S(1))*acos(x_*WC('c', S(1))))**n_, x_), CustomConstraint(lambda a, x: FreeQ(a, x)), CustomConstraint(lambda b, x: FreeQ(b, x)), CustomConstraint(lambda c, x: FreeQ(c, x)), CustomConstraint(lambda d, x: FreeQ(d, x)), CustomConstraint(lambda e, x: FreeQ(e, x)), CustomConstraint(lambda f, x: FreeQ(f, x)), CustomConstraint(lambda e, c, d: ZeroQ(c**S(2)*d + e)), CustomConstraint(lambda n: RationalQ(n)), CustomConstraint(lambda n: Less(n, S(-1))), CustomConstraint(lambda m: IntegerQ(m)), CustomConstraint(lambda m: Greater(m, S(-3))), CustomConstraint(lambda p: PositiveIntegerQ(S(2)*p)))
    rule102 = ReplacementRule(pattern102, lambda b, e, c, m, d, n, a, f, p, x : -c*d**IntPart(p)*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*(m + S(2)*p + S(1))*Int((f*x)**(m + S(1))*(a + b*acos(c*x))**(n + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(b*f*(n + S(1))) + d**IntPart(p)*f*m*(d + e*x**S(2))**FracPart(p)*(-c**S(2)*x**S(2) + S(1))**(-FracPart(p))*Int((f*x)**(m + S(-1))*(a + b*acos(c*x))**(n + S(1))*(-c**S(2)*x**S(2) + S(1))**(p + S(-1)/2), x)/(b*c*(n + S(1))) - (f*x)**m*(a + b*acos(c*x))**(n + S(1))*(d + e*x**S(2))**p*sqrt(-c**S(2)*x**S(2) + S(1))/(b*c*(n + S(1))))
    rubi.add(rule102)
    
    return rubi
"""


def get_skeleton(raw_code, keep_constant: bool = True):
    try:
        tree = cst.parse_module(raw_code)
    except:
        return raw_code

    transformer = CompressTransformer(keep_constant=keep_constant)
    modified_tree = tree.visit(transformer)
    code = modified_tree.code
    code = code.replace(CompressTransformer.replacement_string + "\n", "...\n")
    code = code.replace(CompressTransformer.replacement_string, "...\n")
    return code


# 新增函数，将过长的函数进行缩减，只展示函数的开头100行和结尾100行
def get_skeleton_function(raw_code):
    # 不需要像get_skeleton那样，函数的缩减，是针对函数的，而不是针对模块的
    # 先获取函数的前100行，后获取行数的后100行，然后中间用...表示省略，最后再拼接起来
    if len(raw_code.split("\n")) > 400:
        # 获取前100行
        first_100_lines = "\n".join(raw_code.split("\n")[:50])
        # 获取后100行
        last_100_lines = "\n".join(raw_code.split("\n")[-50:])
        # 拼接
        skeleton =first_100_lines + "\n" + "..." + "\n"
        skeleton += last_100_lines
        return skeleton
    return raw_code


def test_compress():
    skeleton = get_skeleton_function(code)
    print(skeleton)


if __name__ == "__main__":
    test_compress()
