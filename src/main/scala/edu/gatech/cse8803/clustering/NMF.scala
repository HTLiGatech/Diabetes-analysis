package edu.gatech.cse8803.clustering

/**
  * @author Hang Su <hangsu@gatech.edu>
  */


import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix


object NMF {

  /**
   * Run NMF clustering 
   * @param V The original non-negative matrix 
   * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
   * @param maxIterations The maximum number of iterations to perform
   * @param convergenceTol The maximum change in error at which convergence occurs.
   * @return two matrixes W and H in RowMatrix and DenseMatrix format respectively 
   */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {

    /**
     * TODO 1: Implement your code here
     * Initialize W, H randomly 
     * Calculate the initial error (Euclidean distance between V and W * H)
     */
    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze))
    var H = BDM.rand[Double](k, V.numCols().toInt)

    val WH = multiply(W,H)

    val errorReduce = V.rows.zip(WH.rows).map(x => toBreezeVector(x._1) :- toBreezeVector(x._2)).map(x=> x:*x).map(y=> sum(y)).reduce(_+_)

    val initialError = errorReduce * 0.5





    /**
     * TODO 2: Implement your code here
     * Iteratively update W, H in a parallel fashion until error falls below the tolerance value 
     * The updating equations are, 
     * H = H.* W^T^V ./ (W^T^W H)
     * W = W.* VH^T^ ./ (W H H^T^)
     */
    var errorLastIter = 0.0
    var count = 0

    var errorThisIter = initialError
    while (count < maxIterations & (errorThisIter-errorLastIter)> convergenceTol){

      errorLastIter = errorThisIter

      val updateWV = computeWTV(W,V) :/ (computeWTV(W,W) * H :+ 2.0e-15)
      H = H :* updateWV
      val multiplyW = dotDiv(multiply(V,H.t), multiply(W,H*H.t))

      W = dotProd(W,multiplyW)

      val updateMat = multiply(W,H)
      errorThisIter = V.rows.zip(updateMat.rows).map(x => toBreezeVector(x._1) :- toBreezeVector(x._2)).map(x=> x:*x).map(y=> sum(y)).reduce(_+_)

      count = count + 1
     }

    (W, H)
  }


  /**  
  * RECOMMENDED: Implement the helper functions if you needed
  * Below are recommended helper functions for matrix manipulation
  * For the implementation of the first three helper functions (with a null return), 
  * you can refer to dotProd and dotDiv whose implementation are provided
  */
  /**
  * Note:You can find some helper functions to convert vectors and matrices
  * from breeze library to mllib library and vice versa in package.scala
  */

  /** compute the mutiplication of a RowMatrix and a dense matrix */
  def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {

    val result = X.multiply(fromBreeze(d))
    result
    
  }

 /** get the dense matrix representation for a RowMatrix */
  def getDenseMatrix(X: RowMatrix): BDM[Double] = {
    val data = X.rows.map( _.toArray ).collect().flatten
    val rows = X.numRows()
    val cols = X.numCols()

    val denseMat = new BDM(cols.toInt, rows.toInt, data).t
    denseMat
  }

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {

    val result = W.rows.zip(V.rows).map{x=>
      val WT = new BDM[Double](x._1.size, 1, x._1.toArray)
      val V = new BDM[Double](1, x._2.size, x._2.toArray)

      val product = WT * V
      (product)
    }

    result.reduce(_+_)

  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }
}
