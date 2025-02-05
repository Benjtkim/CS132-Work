import numpy as np
import warnings

def swapRows(A, i, j):
    """
    interchange two rows of A
    operates on A in place
    """
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    """
    compute the relative error of a and b
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    """
    reduce row j using row i with pivot pivot, in matrix A
    operates on A in place
    """
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        if np.isclose(A[j][k], factor * A[i][k]):
            A[j][k] = 0.0
        else:
            A[j][k] = A[j][k] - factor * A[i][k]


# stage 1 (forward elimination)
def forwardElimination(B):
    """
    Return the row echelon form of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m-1):
        # Let lefmostNonZeroCol be the position of the leftmost nonzero value 
        # in row i or any row below it 
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        ## for each row below row i (including row i)
        for h in range(i,m):
            ## search, starting from the left, for the first nonzero
            for k in range(i,n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        # if there is no such position, stop
        if leftmostNonZeroRow == m:
            break
        # If the leftmostNonZeroCol in row i is zero, swap this row 
        # with a row below it
        # to make that position nonzero. This creates a pivot in that position.
        if (leftmostNonZeroRow > i):
            swapRows(A, leftmostNonZeroRow, i)
        # Use row reduction operations to create zeros in all positions 
        # below the pivot.
        for h in range(i+1,m):
            rowReduce(A, i, h, leftmostNonZeroCol)
    return A

#################### 

# If any operation creates a row that is all zeros except the last element,
# the system is inconsistent; stop.
def inconsistentSystem(A):
    """
    B is assumed to be in echelon form; return True if it represents
    an inconsistent system, and False otherwise
    """
    nonZeroes = np.nonzero(A)

    #Iterate through the array that has all the column indicies of the non-zeroes
    #in A.
    for i in range(len(nonZeroes[1])):

        #If i is 0 and the first element in the aforementioned array refers to the 
        #last column in the matrix, A is inconsistent.
        if (i == 0) and (nonZeroes[1][i] == len(A[0]) - 1):
            return True
        
        #On the other hand, if the column the current element in the array refers to is the last 
        #column and the element right before also refers to the last column, A is inconsistent.
        elif (nonZeroes[1][i] == len(A[0]) - 1) and (nonZeroes[1][i - 1] == len(A[0]) - 1):
            return True
    
    return False

def backsubstitution(B):

    #Copy matrix B onto matrix A.
    A = B.copy().astype(float)

    #m and n will be the dimensions of A.
    m, n = np.shape(A)

    #For each row starting from the last and working up,
    for i in range(m - 1, -1 , -1):

        #Find the pivot column using np.nonzero. Use try/except because if there's an error, that means 
        # there's no non-zero element in that row.
        try:
            pivotCol = np.nonzero(A[i])[0][0]
        except:
            continue

        #Use row reduction operations to create 0s in all the positions above the pivot.
        for h in range(i - 1, -1 , -1):
            rowReduce(A, i, h, pivotCol)

    #For each row, this nested for loop will search each number until it has 
    #found the first non-zero number. Up until that point, it will divide each 
    #number it comes across by 1. When it has found the first non-zero number, 
    #it will set divideNum to it and foundNum to True so that it can 
    #go on dividing by that number instead of 1. The variables divideNum and 
    #foundNum will reset to 1 and False respectively upon the completion of a 
    #row.
    for i in range(m):
        divideNum = 1.0
        foundNum = False
        for j in range(n):
            if (A[i][j] != 0.0) and (j != n - 1) and (foundNum == False):
                foundNum = True
                divideNum = A[i][j]
                A[i][j] /= divideNum
            else:
                A[i][j] /= divideNum
                
    return A

            






#####################

if __name__ == '__main__':

    A = np.loadtxt('h2m5.txt')
    A = forwardElimination(A)
    print(A)
    print(np.nonzero(A)[0])
    print(np.nonzero(A)[1])
    print(inconsistentSystem(A))
    A = backsubstitution(A)
    print(A)

