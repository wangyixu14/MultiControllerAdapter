def p2c(py_b):
    str_b = str(py_b)
    c_b = str_b.replace("**","^")
    # c_b = c_b.replace("x0","(3*x)")
    # c_b = c_b.replace("x1","(3*y)")    
    return c_b

def trans(py_b):
	str_b = str(py_b)
	c_b = str_b.replace("x","(x/3)")
	c_b = c_b.replace("y","(y/3)")
	c_b = c_b.replace("^","**")
	return c_b

print(p2c('-319.935817838493*(0.5 - 0.25*x1)*(1 - 0.5*x0)**5*(0.5*x1 + 1)**4 - 399.946957257211*(0.5 - 0.25*x1)*(1 - 0.5*x0)**4*(2.0*x0 - 3.0)*(0.5*x1 + 1)**4 - 1799.84218978684*(0.5 - 0.25*x1)*(1 - 0.5*x0)**3*(0.666666666666667*x0 - 1)**2*(0.5*x1 + 1)**4 - 1349.92174908893*(0.5 - 0.25*x1)*(1 - 0.5*x0)**2*(0.666666666666667*x0 - 1)**3*(0.5*x1 + 1)**4 - 126.557612017576*(0.5 - 0.25*x1)*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.5*x1 + 1)**4 - 75.9352716888999*(0.5 - 0.25*x1)*(0.666666666666667*x0 - 1)**5*(0.5*x1 + 1)**4 + 23.6192536126727*(1 - 0.5*x0)**5*(1 - 0.5*x1)**5 + 11.7033615728635*(1 - 0.5*x0)**5*(1 - 0.5*x1)**4*(0.25*x1 + 0.5) - 107.394323437658*(1 - 0.5*x0)**5*(1 - 0.5*x1)**3*(0.5*x1 + 1)**2 - 315.434644447754*(1 - 0.5*x0)**5*(1 - 0.5*x1)**2*(0.5*x1 + 1)**3 - 31.9998485995974*(1 - 0.5*x0)**5*(0.5*x1 + 1)**5 + 28.0544239822634*(1 - 0.5*x0)**4*(1 - 0.5*x1)**5*(2.0*x0 - 3.0) - 5.01589388389946*(1 - 0.5*x0)**4*(1 - 0.5*x1)**4*(2.0*x0 - 3.0)*(0.25*x1 + 0.5) - 176.623122799601*(1 - 0.5*x0)**4*(1 - 0.5*x1)**3*(2.0*x0 - 3.0)*(0.5*x1 + 1)**2 - 395.656673749074*(1 - 0.5*x0)**4*(1 - 0.5*x1)**2*(2.0*x0 - 3.0)*(0.5*x1 + 1)**3 - 39.9998646034803*(1 - 0.5*x0)**4*(2.0*x0 - 3.0)*(0.5*x1 + 1)**5 + 118.8849731783*(1 - 0.5*x0)**3*(1 - 0.5*x1)**5*(0.666666666666667*x0 - 1)**2 - 110.865682887353*(1 - 0.5*x0)**3*(1 - 0.5*x1)**4*(0.666666666666667*x0 - 1)**2*(0.25*x1 + 0.5) - 973.470419210769*(1 - 0.5*x0)**3*(1 - 0.5*x1)**3*(0.666666666666667*x0 - 1)**2*(0.5*x1 + 1)**2 - 1785.13054391813*(1 - 0.5*x0)**3*(1 - 0.5*x1)**2*(0.666666666666667*x0 - 1)**2*(0.5*x1 + 1)**3 - 179.999564096466*(1 - 0.5*x0)**3*(0.666666666666667*x0 - 1)**2*(0.5*x1 + 1)**5 + 83.0589039999059*(1 - 0.5*x0)**2*(1 - 0.5*x1)**5*(0.666666666666667*x0 - 1)**3 - 148.970644569484*(1 - 0.5*x0)**2*(1 - 0.5*x1)**4*(0.666666666666667*x0 - 1)**3*(0.25*x1 + 0.5) - 848.162042751341*(1 - 0.5*x0)**2*(1 - 0.5*x1)**3*(0.666666666666667*x0 - 1)**3*(0.5*x1 + 1)**2 - 1341.51834081319*(1 - 0.5*x0)**2*(1 - 0.5*x1)**2*(0.666666666666667*x0 - 1)**3*(0.5*x1 + 1)**3 - 134.999783623425*(1 - 0.5*x0)**2*(0.666666666666667*x0 - 1)**3*(0.5*x1 + 1)**5 + 6.63717037659708*(1 - 0.5*x1)**5*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4 + 2.85136700166291*(1 - 0.5*x1)**5*(0.666666666666667*x0 - 1)**5 - 19.8175367927388*(1 - 0.5*x1)**4*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.25*x1 + 0.5) - 15.2992697998482*(1 - 0.5*x1)**4*(0.666666666666667*x0 - 1)**5*(0.25*x1 + 0.5) - 90.0526181076045*(1 - 0.5*x1)**3*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.5*x1 + 1)**2 - 59.1304382190112*(1 - 0.5*x1)**3*(0.666666666666667*x0 - 1)**5*(0.5*x1 + 1)**2 - 125.965508731429*(1 - 0.5*x1)**2*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.5*x1 + 1)**3 - 75.6740213246269*(1 - 0.5*x1)**2*(0.666666666666667*x0 - 1)**5*(0.5*x1 + 1)**3 - 12.6562365690179*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.5*x1 + 1)**5 - 7.59374466437586*(0.666666666666667*x0 - 1)**5*(0.5*x1 + 1)**5'))