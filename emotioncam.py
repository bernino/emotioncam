import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time

# initialising
known_face_encodings = []
known_face_metadata = []
emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass

def register_new_face(face_encoding, face_image):
    known_face_encodings.append(face_encoding)
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
    })


def lookup_known_face(face_encoding):
    metadata = None
    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0:
        return metadata

    # Calculate the face distance between the unknown face and every face on in our known face list
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] < 0.65:
        # If we have a match, look up the metadata we've saved for it
        # the index of face_distances is exactly that of known_face_metadata
        metadata = known_face_metadata[best_match_index]

        # Update the metadata for the face so we can keep track of how recently we have seen this face.
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

        # We'll also keep a total "seen count" that tracks how many times this person has come to the door.
        # That, with a time threshold of x seconds
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(seconds=20):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1

    return metadata

def detect_emotion(face_image):
    # setting up for emotion detection
    # the image needs to be a special array for the neural network to work
    face_imagetest = cv2.resize(face_image, (64,64))
    face_imagetest = cv2.cvtColor(face_imagetest, cv2.COLOR_BGR2GRAY)
    face_imagetest = face_imagetest.astype("float") / 255.0
    face_imagetest = img_to_array(face_imagetest)
    face_imagetest = np.expand_dims(face_imagetest, axis=0)
    predicted_emotions = emotion_classifier.predict(face_imagetest)[0]
    predicted_emotions = predicted_emotions.tolist()
    # the predicted_emotions has the same order as as the dict in init
    return predicted_emotions

def main_loop():
    # lets get a feed
    video_capture = cv2.VideoCapture(0)

    # Track how long since we last saved a copy of our known faces to disk as a backup.
    number_of_faces_since_save = 0

    while True:
        time.sleep(0.5)
        # Grab a single frame of video
        ret, frame = video_capture.read()
        frame = cv2.flip(frame,1)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_labels = []

        for face_location, face_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = face_location
            face_image = small_frame[top:bottom, left:right]
            preds = detect_emotion(face_image)
            np_preds = np.asarray(preds)
            #emotion_probability = np.max(np_preds)
            max_emotion = emotions[np_preds.argmax()]
            print("MAX EMOTION: {}".format(max_emotion))
            for i, e in enumerate(emotions):
                print("{} with prob {}\n".format(e,preds[i]))
            print("\n")

            metadata = lookup_known_face(face_encoding)

            if metadata is not None:
                time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                face_label = max_emotion
                #face_label = f"Seen {int(time_at_door.total_seconds())}s"

            else:
                face_label = "New face!"

                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))

                register_new_face(face_encoding, face_image)

            face_labels.append(face_label)

        # Draw a box around each face and label each face
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display recent visitor images - put at zero since inside while loop
        number_of_recent_visitors = 0
        for metadata in known_face_metadata:
            # If we have seen this person in the last minute, draw their image
            if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 5:
                # Draw the known face image
                x_position = number_of_recent_visitors * 150
                frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                number_of_recent_visitors += 1

                # Label the image with how many times they have visited
                visits = metadata['seen_count']
                visit_label = f"{visits} seen"
                if visits == 1:
                    visit_label = "First seen"
                cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        if number_of_recent_visitors > 0:
            cv2.putText(frame, "Face", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        # save to disk every 500th step
        if len(face_locations) > 0 and number_of_faces_since_save > 500:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_known_faces()
    main_loop()
