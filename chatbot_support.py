import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

SPANISH_STOPWORDS = {
    "a", "al", "algo", "con", "de", "del", "desde", "donde", "durante",
    "el", "ella", "en", "entre", "es", "esta", "hay", "la", "las", "lo",
    "los", "mas", "me", "mi", "no", "nos", "o", "para", "pero", "por",
    "que", "se", "si", "sin", "su", "sus", "un", "una", "y", "ya", "yo"
}

TRAINING_DATA = [
    ("no tengo internet", "conexion"),
    ("wifi no funciona", "conexion"),
    ("internet caido", "conexion"),
    ("no me conecta a la red", "conexion"),
    ("se fue el internet", "conexion"),
    ("mi wifi falla", "conexion"),
    ("no carga ninguna pagina", "conexion"),

    ("olvide mi contrasena", "contrasena"),
    ("no recuerdo mi password", "contrasena"),
    ("quiero cambiar mi contrasena", "contrasena"),
    ("resetear clave", "contrasena"),
    ("no puedo entrar por la contrasena", "contrasena"),
    ("he perdido la clave", "contrasena"),

    ("no puedo acceder al correo", "correo"),
    ("outlook no abre", "correo"),
    ("no me llegan correos", "correo"),
    ("el correo falla", "correo"),
    ("problema con email", "correo"),
    ("mi cuenta de correo no sincroniza", "correo"),

    ("el ordenador va lento", "rendimiento"),
    ("mi pc esta muy lento", "rendimiento"),
    ("el equipo se congela", "rendimiento"),
    ("el ordenador tarda mucho", "rendimiento"),
    ("las aplicaciones van lentas", "rendimiento"),
    ("el sistema tarda en arrancar", "rendimiento"),

    ("la impresora no imprime", "impresora"),
    ("no funciona la impresora", "impresora"),
    ("la impresora no detecta el ordenador", "impresora"),
    ("atasco de papel", "impresora"),
    ("la impresora esta desconectada", "impresora"),
    ("sale error al imprimir", "impresora"),

    ("hola", "saludo"),
    ("buenos dias", "saludo"),
    ("buenas tardes", "saludo"),
    ("hola que tal", "saludo"),

    ("adios", "despedida"),
    ("hasta luego", "despedida"),
    ("gracias adios", "despedida"),
    ("nos vemos", "despedida")
]

RESPONSES = {
    "conexion": (
        "Parece un problema de conexión. Reinicia el router, comprueba que el WiFi "
        "este activado y revisa si el cable de red esta bien conectado. "
        "Si sigue fallando, contacta con soporte tecnico."
    ),
    "contrasena": (
        "Parece un problema de contraseña. Accede al portal de la empresa y pulsa "
        "en la opcion de recuperar u olvidar contraseña. Despues revisa tu correo."
    ),
    "correo": (
        "Se ha detectado una incidencia con el correo. Comprueba la conexión a internet, "
        "reinicia Outlook o la aplicacion correspondiente y revisa la configuracion de la cuenta."
    ),
    "rendimiento": (
        "El problema parece relacionado con el rendimiento del equipo. Reinicia el ordenador, "
        "cierra programas innecesarios, revisa el espacio libre y pasa el antivirus."
    ),
    "impresora": (
        "Parece un fallo de impresora. Revisa que este encendida, que tenga papel o tinta "
        "y que este correctamente conectada al ordenador o a la red."
    ),
    "saludo": (
        "Hola, soy el asistente de soporte tecnico. Puedes escribirme incidencias de conexión, "
        "contraseña, correo, rendimiento o impresora."
    ),
    "despedida": (
        "Hasta luego. Si necesitas mas ayuda, puedes volver a escribirme cuando quieras."
    ),
    "desconocido": (
        "No he podido identificar bien el problema. Intenta explicarlo con otras palabras "
        "o contacta con soporte humano."
    )
}


class SupportChatbot:
    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self):
        self.history = []
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs"
            ))
        ])
        self.test_report = ""
        self.train_and_evaluate()

    def train_and_evaluate(self):
        X = [self._preprocess(phrase) for phrase, _ in TRAINING_DATA]
        y = [intent for _, intent in TRAINING_DATA]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        self.test_report = classification_report(y_test, y_pred, zero_division=0)

        print("\n--- EVALUACION REAL CON TRAIN / TEST SPLIT ---")
        print(f"Frases de entrenamiento: {len(X_train)}")
        print(f"Frases de prueba: {len(X_test)}")
        print(f"Accuracy en test: {accuracy:.3f}")
        print(self.test_report)

        self.pipeline.fit(X, y)
        self.classes = list(self.pipeline.classes_)

    def _preprocess(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in SPANISH_STOPWORDS and len(t) > 1]
        return " ".join(tokens)

    def predict_intent(self, text: str):
        processed_text = self._preprocess(text)
        intent = self.pipeline.predict([processed_text])[0]
        confidence = max(self.pipeline.predict_proba([processed_text])[0])

        if confidence < self.CONFIDENCE_THRESHOLD:
            intent = "desconocido"

        return intent, round(confidence, 3)

    def respond(self, text: str):
        intent, confidence = self.predict_intent(text)
        response = RESPONSES.get(intent, RESPONSES["desconocido"])

        result = {
            "usuario": text,
            "intencion": intent,
            "confianza": confidence,
            "respuesta": response
        }

        self.history.append(result)
        return result

    def interactive_chat(self):
        print("=== Chatbot de soporte tecnico ===")
        print("Escribe 'salir' para terminar.\n")

        while True:
            text = input("Tu: ").strip()

            if text.lower() == "salir":
                print("Bot: Hasta luego.")
                break

            result = self.respond(text)
            print(f"Bot [{result['intencion']} | confianza={result['confianza']}]:")
            print(result["respuesta"])
            print()


if __name__ == "__main__":
    chatbot = SupportChatbot()
    chatbot.interactive_chat()
