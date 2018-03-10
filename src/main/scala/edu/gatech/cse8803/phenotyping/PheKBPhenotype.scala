/**
  * @author Hang Su <hangsu@gatech.edu>,
  * @author Sungtae An <stan84@gatech.edu>,
  */

package edu.gatech.cse8803.phenotyping

import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import org.apache.spark.rdd.RDD

object T2dmPhenotype {
  
  // criteria codes given
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
      "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
      "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
      "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
      "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
      "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
      "avandia", "actos", "actos", "glipizide")

  /**
    * Transform given data set to a RDD of patients and corresponding phenotype
    * @param medication medication RDD
    * @param labResult lab result RDD
    * @param diagnostic diagnostic code RDD
    * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
    */

  def abnormal(lab: LabResult): Boolean = {
    lab.testName.toLowerCase() match {
      case "hba1c" => lab.value >= 6
      case "hemoglobin a1c" => lab.value >= 6
      case "fasting glucose" => lab.value >= 110
      case "fasting blood glucose" => lab.value >= 110
      case "fasting plasma glucose" => lab.value >= 110
      case "glucose" => lab.value >= 110
      case "glucose, serum" => lab.value >= 110
      case   _=> false
    }
  }

  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
      * Remove the place holder and implement your code here.
      * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
      * When testing your code, we expect your function to have no side effect,
      * i.e. do NOT read from file or write file
      *
      * You don't need to follow the example placeholder code below exactly, but do have the same return type.
      *
      * Hint: Consider case sensitivity when doing string comparisons.
      */

    val sc = medication.sparkContext

    /** Hard code the criteria */
    // val type1_dm_dx = Set("code1", "250.03")
    // val type1_dm_med = Set("med1", "insulin nph")
    // use the given criteria above like T1DM_DX, T2DM_DX, T1DM_MED, T2DM_MED and hard code DM_RELATED_DX criteria as well
    val type1_dm_dx = T1DM_DX
    val type1_dm_med = T1DM_MED
    val type2_dm_dx = T2DM_DX
    val type2_dm_med = T2DM_MED

    /** Find CASE Patients */
    val filteredDiag = diagnostic.filter(line => !type1_dm_dx.contains(line.code) && type2_dm_dx.contains(line.code))
    val filteredID = filteredDiag.map(x => x.patientID).distinct()

    val medType1 = medication.filter(line => type1_dm_med.contains(line.medicine.toLowerCase))
    val medType2 = medication.filter(line => type2_dm_med.contains(line.medicine.toLowerCase))

    val condition1 = filteredID.subtract(medType1.map(line=>line.patientID))
    val condition2 = filteredID.intersection(medType1.map(line=>line.patientID)).subtract(medType2.map(line=>line.patientID))

    val med1DateSort = medType1.groupBy(line => line.patientID).map(x => (x._1,x._2.minBy(y => y.date).date))
    val med2DateSort = medType2.groupBy(line => line.patientID).map(x => (x._1,x._2.minBy(y => y.date).date))
    val before = med2DateSort.join(med1DateSort).filter(line=>line._2._1.before(line._2._2)).map(x=>x._1)

    val condition3 = filteredID.intersection(before)

    val casePatients = condition1.union(condition2).union(condition3).distinct().map(x => (x,1))

    /** Find CONTROL Patients */
    val contailGlucose = labResult.filter(line => (line.testName.toLowerCase.contains("glucose")))

    val notInAbnormal = contailGlucose.filter(line => !abnormal(line)).map(x=>x.patientID)

    val DM_RELATED_DX = Set("790.21","790.22","790.2","790.29","648.81","648.82","648.83","648.84","648","648.01","648.02","648.03","648.04","791.5","277.7","V77.1","256.4")

    val DBCases = diagnostic.filter(line => DM_RELATED_DX.contains(line.code) || line.code.contains("250.")).map(line => line.patientID)

    val controlCases = notInAbnormal.subtract(DBCases)

    val controlPatients = controlCases.distinct().map(line=> (line,2))


    /** Find OTHER Patients */
    val AllPatients = medication.map(line => line.patientID).union(labResult.map(line => line.patientID)).union(diagnostic.map(line => line.patientID))
    val others = AllPatients.subtract(casePatients.map(line => line._1)).subtract(controlPatients.map(line => line._1)).map(line => (line, 3))

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients, controlPatients, others)

    /** Return */
    phenotypeLabel
  }
}
