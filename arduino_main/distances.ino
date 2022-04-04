const unsigned int BAUD_RATE=9600;

const unsigned int TRIG_PIN_FRONT=2;
const unsigned int TRIG_PIN_RIGHT=3;
const unsigned int TRIG_PIN_BACK=4;
const unsigned int TRIG_PIN_LEFT=5;

const unsigned int ECHO_PIN_FRONT=6;
const unsigned int ECHO_PIN_RIGHT=7;
const unsigned int ECHO_PIN_BACK=8;
const unsigned int ECHO_PIN_LEFT=9;

float pwm_value;
float angle = 0;

void setup() {
  //Initialisation des triggers
  pinMode(TRIG_PIN_FRONT, OUTPUT);
  pinMode(TRIG_PIN_RIGHT, OUTPUT);
  pinMode(TRIG_PIN_BACK, OUTPUT);
  pinMode(TRIG_PIN_LEFT, OUTPUT);

  //Initialisation des echos
  pinMode(ECHO_PIN_FRONT, INPUT);
  pinMode(ECHO_PIN_RIGHT, INPUT);
  pinMode(ECHO_PIN_BACK, INPUT);
  pinMode(ECHO_PIN_LEFT, INPUT);

  //Initialisation de la communication série
  Serial.begin(BAUD_RATE);
  Serial.setTimeout(1);
}

void loop() {
    //Envoi du signal de trigger pour l'avant
    digitalWrite(TRIG_PIN_FRONT, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN_FRONT, LOW);
  
    //Récupération de la distance devant
    const unsigned long duration_front = pulseIn(ECHO_PIN_FRONT, HIGH);
    int distance_front = duration_front/29/2;

    //Envoi du signal de trigger pour la droite
    digitalWrite(TRIG_PIN_RIGHT, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN_RIGHT, LOW);

    //Récupération de la distance à droite
    const unsigned long duration_right = pulseIn(ECHO_PIN_RIGHT, HIGH);
    int distance_right = duration_right/29/2;

    //Envoi du signal de trigger pour l'arrière
    digitalWrite(TRIG_PIN_BACK, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN_BACK, LOW);
  
    //Récupération de la distance devant
    const unsigned long duration_back = pulseIn(ECHO_PIN_BACK, HIGH);
    int distance_back = duration_back/29/2;


    //Envoi du signal de trigger pour la droite
    digitalWrite(TRIG_PIN_LEFT, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN_LEFT, LOW);

    //Récupération de la distance à droite
    const unsigned long duration_left = pulseIn(ECHO_PIN_LEFT, HIGH);
    int distance_left = duration_left/29/2;

    delay(100);

    //Envoi des distances dans les 4 directions
    Serial.print(String(String(distance_front)+","+String(distance_right)+","+String(distance_back)+","+String(distance_left))+"\n");
}
