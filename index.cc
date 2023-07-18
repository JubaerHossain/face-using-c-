#include <iostream>
#include <mysql_driver.h>
#include <mysql_connection.h>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>

using namespace std;
using namespace cv;
using namespace dlib;

// Employee structure to store employee data
struct Employee {
    int id;
    string name;
    vector<Mat> faceImages;
};

int main() {
    // Connect to the MySQL database
    sql::mysql::MySQL_Driver *driver;
    sql::Connection *con;

    driver = sql::mysql::get_mysql_driver_instance();
    con = driver->connect("tcp://127.0.0.1:3306", "username", "password");
    con->setSchema("database_name");

    // Load face detection model
    frontal_face_detector detector = get_frontal_face_detector();

    // Load face recognition model
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    // Load employee data from MySQL
    sql::Statement *stmt;
    sql::ResultSet *res;

    stmt = con->createStatement();
    res = stmt->executeQuery("SELECT id, name FROM employees");

    // Populate employee data
    vector<Employee> employees;

    while (res->next()) {
        Employee employee;
        employee.id = res->getInt("id");
        employee.name = res->getString("name");
        employees.push_back(employee);
    }

    delete res;
    delete stmt;

    // Load employee face images
    for (Employee& employee : employees) {
        // Assuming face images are stored in a directory named after the employee ID
        string imageDirectory = "employee_images/" + to_string(employee.id) + "/";
        
        // Load and store face images for the employee
        for (int i = 1; i <= 3; ++i) {
            string imagePath = imageDirectory + "image" + to_string(i) + ".jpg";
            Mat faceImage = imread(imagePath);
            
            // Add the face image to the employee's faceImages vector
            employee.faceImages.push_back(faceImage);
        }
    }

    // Load input image
    Mat inputImage = imread("input_image.jpg");
    if (inputImage.empty()) {
        cerr << "Failed to load input image." << endl;
        return -1;
    }

    // Convert input image to grayscale
    Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

    // Convert OpenCV Mat to Dlib matrix
    cv_image<unsigned char> dlibImage(grayImage);

    // Detect faces in the input image
    std::vector<rectangle> faceRectangles = detector(dlibImage);

    // Process each detected face
    for (rectangle faceRectangle : faceRectangles) {
        // Convert face region to an OpenCV Rect
        Rect cvFaceRect(faceRectangle.left(), faceRectangle.top(), 
                        faceRectangle.width(), faceRectangle.height());

        // Crop face region from the input image
        Mat croppedFace = inputImage(cvFaceRect).clone();

        // Perform face recognition for each employee
        for (Employee& employee : employees) {
            // Calculate similarity scores for each stored face image of the employee
            for (const Mat& storedFace : employee.faceImages) {
                // Perform face recognition on the stored face image

                // Convert stored face image to grayscale
                Mat grayStoredFace;
                cvtColor(storedFace, grayStoredFace, COLOR_BGR2GRAY);

                // Convert OpenCV Mat to Dlib matrix
                cv_image<unsigned char> dlibStoredFace(grayStoredFace);

                // Detect landmarks for the stored face image
                full_object_detection landmarks = sp(dlibStoredFace, faceRectangle);

                // Perform face recognition using the stored face image and landmarks
                matrix<float, 0, 1> storedFaceEmbedding = net(dlibStoredFace, landmarks);

                // Calculate similarity score between the stored face embedding and the input face
                float similarityScore = length(storedFaceEmbedding - inputFaceEmbedding);

                // Set a threshold for recognition (you may need to experiment with this value)
                float recognitionThreshold = 0.6f;

                if (similarityScore < recognitionThreshold) {
                    // Recognized as an employee
                    cout << "Employee ID: " << employee.id << endl;
                    cout << "Employee Name: " << employee.name << endl;

                    // Store attendance record in MySQL
                    sql::PreparedStatement *prepStmt;
                    prepStmt = con->prepareStatement("INSERT INTO attendance (employee_id, timestamp, status) VALUES (?, ?, ?)");

                    // Get current timestamp
                    time_t now = time(nullptr);
                    string timestamp = to_string(now);

                    // Insert attendance record into the database
                    prepStmt->setInt(1, employee.id);
                    prepStmt->setString(2, timestamp);
                    prepStmt->setString(3, "check-in");
                    prepStmt->executeUpdate();

                    delete prepStmt;

                    // Break out of the loop if the employee is recognized
                    break;
                }
            }
        }
    }

    // Close the database connection
    delete con;

    return 0;
}
