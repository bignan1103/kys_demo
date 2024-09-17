package com.kys.demo

interface SimilarityClassifier {
    /** An immutable result returned by a Classifier describing what was recognized.  */
    class Recognition(
        private val id: String?,
        /** Display name for the recognition.  */
        private val title: String?, private val distance: Float?
    ) {
        var extra: Any? = null

        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }

            if (title != null) {
                resultString += "$title "
            }

            if (distance != null) {
                resultString += "(%.1f%%) ".format(distance * 100.0f)
            }

            return resultString.trim { it <= ' ' }
        }
    }
}