"""Separate camera-free package: test actual app model loader without replacing AppHear."""
from pathlib import Path
import hashlib,json

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
APP=ROOT/'android/app/src/main/java/com/example/apphear'
OUT=RUN/'phone_parity'
assert not OUT.exists(),'HARNESS_ALREADY_PREPARED'
OUT.mkdir()
def write(name,text):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
write('settings.gradle.kts','''pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS); repositories { google(); mavenCentral() } }
rootProject.name = "AppHearContinuousModelParity"
include(":app")
''')
write('build.gradle.kts','''plugins {
 id("com.android.application") version "8.11.2" apply false
 id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}
''')
assets=(ROOT/'packages/continuous_asl_research_01/android_assets').as_posix()
fixtures=(ROOT/'packages/continuous_asl_research_01/parity_assets').as_posix()
write('app/build.gradle.kts','''plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }
android {
 namespace = "com.example.apphear"
 compileSdk = 36
 defaultConfig {
  applicationId = "com.example.apphear.continuousparity"
  minSdk = 26
  targetSdk = 36
  versionCode = 1
  versionName = "1"
  testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  buildConfigField("boolean", "CONTINUOUS_ASL_EXPERIMENT_ENABLED", "true")
 }
 buildFeatures { buildConfig = true }
 compileOptions { sourceCompatibility = JavaVersion.VERSION_11; targetCompatibility = JavaVersion.VERSION_11 }
 kotlinOptions { jvmTarget = "11" }
 sourceSets.getByName("main").assets.srcDir("'''+assets+'''")
 sourceSets.getByName("androidTest").assets.srcDir("'''+fixtures+'''")
 androidResources { noCompress += "onnx" }
}
dependencies {
 implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
 androidTestImplementation("androidx.test.ext:junit:1.3.0")
 androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
}
''')
write('app/src/main/AndroidManifest.xml','''<manifest xmlns:android="http://schemas.android.com/apk/res/android"><application android:label="AppHear model parity" android:allowBackup="false" /></manifest>''')
write('local.properties','sdk.dir=C:/Users/johnm/AppData/Local/Android/Sdk\n')
model=APP/'continuousasl/ContinuousAslModel.kt'
write('app/src/main/java/com/example/apphear/continuousasl/ContinuousAslModel.kt',model.read_text())
core=(APP/'continuousasl/ContinuousAslCore.kt').read_text()
policy=core[core.index('internal data class ContinuousAslPolicy'):core.index('internal enum class ContinuousAslPhase')]
write('app/src/main/java/com/example/apphear/continuousasl/Policy.kt','package com.example.apphear.continuousasl\n\n'+policy)
# Only the contract constants are needed for MODEL parity; this harness never
# claims camera/landmark/feature-builder parity and requests no device permissions.
write('app/src/main/java/com/example/apphear/perceptionvnext/Contract.kt','''package com.example.apphear.perceptionvnext
internal object PhoneSegmentationFeatureBuilder {
 const val SCHEMA = "apphear.perception_vnext.segmentation_features.v1"
 const val FEATURE_WIDTH = 378
}
''')
test=ROOT/'android/app/src/androidTest/java/com/example/apphear/continuousasl/ContinuousAslParityInstrumentedTest.kt'
write('app/src/androidTest/java/com/example/apphear/continuousasl/ContinuousAslParityInstrumentedTest.kt',test.read_text())
write('app/src/androidTest/java/com/example/apphear/continuousasl/ModelTimingTest.kt','''package com.example.apphear.continuousasl
import android.os.Bundle
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.json.JSONArray
import org.junit.Test
import org.junit.Assert.assertTrue
class ModelTimingTest {
 @Test fun boundedSyntheticCpuTiming() {
  val instrumentation = InstrumentationRegistry.getInstrumentation()
  val rows = JSONArray()
  ContinuousAslModel(instrumentation.targetContext).use { model ->
   for (length in listOf(1,17,61)) {
    val frames = (0 until length).map { t -> FloatArray(378) { f -> ((t+f)%17-8)*0.125f } }
    repeat(3) { model.predictLast(frames) }
    val times = (0 until 20).map {
     val start=SystemClock.elapsedRealtimeNanos()
     val token=model.predictLast(frames)
     assertTrue(token in model.vocabulary.indices)
     (SystemClock.elapsedRealtimeNanos()-start)/1e6
    }.sorted()
    rows.put(JSONObject().put("observations",length).put("runs",20)
     .put("p50_ms",(times[9]+times[10])/2).put("p95_ms",times[18]).put("max_ms",times.last()))
   }
  }
  val output=JSONObject().put("scope","SYNTHETIC_CPU_MODEL_ONLY_NOT_CAMERA_LATENCY")
   .put("device",android.os.Build.MODEL).put("results",rows)
  instrumentation.sendStatus(2,Bundle().apply { putString("stream","APPHEAR_MODEL_TIMING="+output.toString()+"\\n") })
 }
}
''')
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [model,test]}
write('harness_provenance.json',json.dumps({'purpose':'Existing AppHear packaged epoch12 model parity and synthetic CPU timing; NOT the192 checkpoint','separate_application_id':'com.example.apphear.continuousparity','source_hashes':sources,'camera_or_audio_permissions':False,'feature_math_tested':False,'installed_apphear_replaced':False},indent=2))
print(OUT)
