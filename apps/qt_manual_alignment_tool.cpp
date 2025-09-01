#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QCheckBox>
#include <QListWidget>
#include <QScrollArea>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QRadioButton>
#include <QGroupBox>
#include <QTextEdit>
#include <QMessageBox>
#include <QTextStream>

#include <QGridLayout>
#include <QButtonGroup>
#include <QDebug>
#include <QTime>
#include <QFileInfo>

#include <QVTKOpenGLNativeWidget.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include "color_icp/color_icp.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>

#include <memory>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

class PointCloudViewer : public QMainWindow {
    Q_OBJECT

private:
    // Cloud selection state
    int originCloudIndex = -1;
    int movingCloudIndex = -1;
    bool uniqueColorMode = true;
    bool showOriginalColors = false;

    struct CloudData {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
        cv::Mat transform_matrix;         // Current transform
        cv::Mat original_transform_matrix; // Initial transform
        std::string cloud_name;
        
        // Store DOF adjustments
        std::array<float, 6> dof_adjustments = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // TX, TY, TZ, RX, RY, RZ
        
        // VTK visualization objects
        vtkSmartPointer<vtkActor> actor;
        vtkSmartPointer<vtkPolyData> polydata;
        vtkSmartPointer<vtkPolyDataMapper> mapper;

        CloudData() : cloud(new pcl::PointCloud<pcl::PointXYZRGB>),
                     transform_matrix(cv::Mat::eye(4, 4, CV_32F)),
                     original_transform_matrix(cv::Mat::eye(4, 4, CV_32F)) {}
    };

    enum class TransformMode {
        TRANSLATE_X, TRANSLATE_Y, TRANSLATE_Z,
        ROTATE_X, ROTATE_Y, ROTATE_Z
    };

public:
    PointCloudViewer(QWidget *parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Pointcloud Aligner");
        
        // Create the main widget and layout
        QWidget* centralWidget = new QWidget(this);
        QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
        
        // Create VTK widget
        qvtkWidget = new QVTKOpenGLNativeWidget();
        
        // Create renderer
        renderer = vtkSmartPointer<vtkRenderer>::New();
        renderer->SetBackground(0.0, 0.0, 0.0);  // Black background
        
        // Get the render window and add our renderer
        renderWindow = qvtkWidget->renderWindow();
        renderWindow->AddRenderer(renderer);
        
        // Set up camera interaction style
        vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = 
            vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
        qvtkWidget->interactor()->SetInteractorStyle(style);
        
        // Create orientation marker (axes)
        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
        orientationWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
        orientationWidget->SetOutlineColor(0.9300, 0.5700, 0.1300);
        orientationWidget->SetOrientationMarker(axes);
        orientationWidget->SetInteractor(qvtkWidget->interactor());
        orientationWidget->SetViewport(0.0, 0.0, 0.2, 0.2);
        orientationWidget->SetEnabled(1);
        orientationWidget->InteractiveOff();
        
        // Set black background
        renderer->SetBackground(0.0, 0.0, 0.0);
        
        // Create right panel with scroll area for controls
        QScrollArea* scrollArea = new QScrollArea();
        QWidget* rightPanel = new QWidget();
        QVBoxLayout* rightLayout = new QVBoxLayout(rightPanel);
        
        // Cloud Loading Section
        QGroupBox* loadGroup = new QGroupBox("Point Clouds");
        QVBoxLayout* loadLayout = new QVBoxLayout(loadGroup);
        
        QPushButton* loadButton = new QPushButton("Load Point Cloud", loadGroup);
        connect(loadButton, &QPushButton::clicked, this, &PointCloudViewer::loadPointCloud);
        
        cloudList = new QListWidget(loadGroup);
        loadLayout->addWidget(loadButton);
        loadLayout->addWidget(cloudList);
        
        // Origin Cloud Selection
        QGroupBox* originGroup = new QGroupBox("Origin Cloud Selection");
        QVBoxLayout* originLayout = new QVBoxLayout(originGroup);
        
        QLabel* originInstruction = new QLabel("Select the reference/origin cloud:");
        originInstruction->setStyleSheet("QLabel { font-style: italic; color: #666666; }");
        originLayout->addWidget(originInstruction);
        
        originCloudList = new QListWidget(originGroup);
        originLayout->addWidget(originCloudList);
        
        // Moving Cloud Selection
        QGroupBox* movingGroup = new QGroupBox("Moving Cloud Selection");
        QVBoxLayout* movingLayout = new QVBoxLayout(movingGroup);
        
        QLabel* movingInstruction = new QLabel("Select the cloud to transform:");
        movingInstruction->setStyleSheet("QLabel { font-style: italic; color: #666666; }");
        movingLayout->addWidget(movingInstruction);
        
        movingCloudList = new QListWidget(movingGroup);
        movingLayout->addWidget(movingCloudList);
        
        // Transform Control Section
        QGroupBox* transformGroup = new QGroupBox("Transform Controls");
        QVBoxLayout* transformLayout = new QVBoxLayout(transformGroup);
        
        // Step size control
        QHBoxLayout* stepLayout = new QHBoxLayout();
        QLabel* stepLabel = new QLabel("Step Size:");
        stepSizeSpinBox = new QDoubleSpinBox();
        stepSizeSpinBox->setRange(0.001, 1.0);
        stepSizeSpinBox->setValue(0.01);
        stepSizeSpinBox->setSingleStep(0.001);
        stepLayout->addWidget(stepLabel);
        stepLayout->addWidget(stepSizeSpinBox);
        
        // Transform buttons grid
        QGridLayout* transformButtonLayout = new QGridLayout();
        
        // ColorICP button
        QPushButton* colorICPButton = new QPushButton("Refine with ColorICP");
        transformLayout->addWidget(colorICPButton);
        connect(colorICPButton, &QPushButton::clicked, this, &PointCloudViewer::refineWithColorICP);
        
        // Translation buttons
        tx_minus_btn = new QPushButton("-X");
        tx_plus_btn = new QPushButton("+X");
        ty_minus_btn = new QPushButton("-Y");
        ty_plus_btn = new QPushButton("+Y");
        tz_minus_btn = new QPushButton("-Z");
        tz_plus_btn = new QPushButton("+Z");
        
        transformButtonLayout->addWidget(tx_minus_btn, 0, 0);
        transformButtonLayout->addWidget(tx_plus_btn, 0, 1);
        transformButtonLayout->addWidget(ty_minus_btn, 1, 0);
        transformButtonLayout->addWidget(ty_plus_btn, 1, 1);
        transformButtonLayout->addWidget(tz_minus_btn, 2, 0);
        transformButtonLayout->addWidget(tz_plus_btn, 2, 1);
        
        // Rotation buttons
        rx_minus_btn = new QPushButton("-RX");
        rx_plus_btn = new QPushButton("+RX");
        ry_minus_btn = new QPushButton("-RY");
        ry_plus_btn = new QPushButton("+RY");
        rz_minus_btn = new QPushButton("-RZ");
        rz_plus_btn = new QPushButton("+RZ");
        
        transformButtonLayout->addWidget(rx_minus_btn, 0, 2);
        transformButtonLayout->addWidget(rx_plus_btn, 0, 3);
        transformButtonLayout->addWidget(ry_minus_btn, 1, 2);
        transformButtonLayout->addWidget(ry_plus_btn, 1, 3);
        transformButtonLayout->addWidget(rz_minus_btn, 2, 2);
        transformButtonLayout->addWidget(rz_plus_btn, 2, 3);
        
        transformLayout->addLayout(stepLayout);
        transformLayout->addLayout(transformButtonLayout);
        
        // Display Options Group
        QGroupBox* displayGroup = new QGroupBox("Display Options");
        QVBoxLayout* displayLayout = new QVBoxLayout(displayGroup);
        
        QCheckBox* uniqueColorCheckbox = new QCheckBox("Unique Color per Cloud");
        uniqueColorCheckbox->setChecked(true);
        uniqueColorCheckbox->setStyleSheet("QCheckBox { font-weight: bold; color: darkblue; }");
        
        QCheckBox* originalColorCheckbox = new QCheckBox("Show Original Point Colors");
        originalColorCheckbox->setChecked(false);
        originalColorCheckbox->setStyleSheet("QCheckBox { font-weight: bold; color: darkgreen; }");
        
        displayLayout->addWidget(uniqueColorCheckbox);
        displayLayout->addWidget(originalColorCheckbox);
        
        // Transform Display
        transformOutput = new QTextEdit();
        transformOutput->setReadOnly(true);
        transformOutput->setMinimumHeight(100);
        transformLayout->addWidget(transformOutput);
        
        // Add all sections to right panel
        rightLayout->addWidget(loadGroup);
        rightLayout->addWidget(originGroup);     // Add origin cloud selection group
        rightLayout->addWidget(movingGroup);     // Add moving cloud selection group
        rightLayout->addWidget(transformGroup);
        rightLayout->addWidget(displayGroup);    // Add display options group
        rightLayout->addStretch();
        
        scrollArea->setWidget(rightPanel);
        scrollArea->setWidgetResizable(true);
        scrollArea->setMinimumWidth(400);
        
        mainLayout->addWidget(qvtkWidget, 4);  // Viewer gets 4/5 of the width
        mainLayout->addWidget(scrollArea, 1);   // Right panel gets 1/5 of the width
        
        setCentralWidget(centralWidget);
        resize(1200, 800);
        
        // Connect transform control signals
        connect(stepSizeSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [this](double value) { stepSize = static_cast<float>(value); });
        
        connect(tx_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_X, -stepSize); });
        connect(tx_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_X, stepSize); });
        connect(ty_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_Y, -stepSize); });
        connect(ty_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_Y, stepSize); });
        connect(tz_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_Z, -stepSize); });
        connect(tz_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_Z, stepSize); });
        
        connect(rx_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_X, -stepSize); });
        connect(rx_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_X, stepSize); });
        connect(ry_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_Y, -stepSize); });
        connect(ry_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_Y, stepSize); });
        connect(rz_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_Z, -stepSize); });
        connect(rz_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_Z, stepSize); });
        
        // Connect display option signals
        connect(uniqueColorCheckbox, &QCheckBox::toggled, [this](bool checked) {
            uniqueColorMode = checked;
            updateCloudColors();
            qvtkWidget->renderWindow()->Render();
        });
        
        connect(originalColorCheckbox, &QCheckBox::toggled, [this](bool checked) {
            showOriginalColors = checked;
            updateCloudColors();
            qvtkWidget->renderWindow()->Render();
        });

        // Connect cloud selection signals
        connect(originCloudList, &QListWidget::currentRowChanged, [this](int row) {
            originCloudIndex = row;
            updateCloudVisuals();
            updateTransformDisplay();
        });
        
        connect(movingCloudList, &QListWidget::currentRowChanged, [this](int row) {
            movingCloudIndex = row;
            updateCloudVisuals();
            updateTransformDisplay();
        });
    }

private slots:
    void refineWithColorICP() {
        if (movingCloudIndex < 0 || originCloudIndex < 0) {
            QMessageBox::warning(this, "Warning", "Please select both origin and moving clouds first.");
            return;
        }

        // Convert CloudData clouds to PCL format with normals
        sb::ColorICP::PointCloudPtr originCloud(new pcl::PointCloud<sb::ColorICP::PointT>());
        sb::ColorICP::PointCloudPtr movingCloud(new pcl::PointCloud<sb::ColorICP::PointT>());
        
        // Convert RGB clouds to RGB+Normal clouds
        for (const auto& p : *clouds[originCloudIndex].cloud) {
            sb::ColorICP::PointT point;
            point.x = p.x;
            point.y = p.y;
            point.z = p.z;
            point.r = p.r;
            point.g = p.g;
            point.b = p.b;
            originCloud->points.push_back(point);
        }
        originCloud->width = originCloud->points.size();
        originCloud->height = 1;
        
        for (const auto& p : *clouds[movingCloudIndex].cloud) {
            sb::ColorICP::PointT point;
            point.x = p.x;
            point.y = p.y;
            point.z = p.z;
            point.r = p.r;
            point.g = p.g;
            point.b = p.b;
            movingCloud->points.push_back(point);
        }
        movingCloud->width = movingCloud->points.size();
        movingCloud->height = 1;

        sb::ColorICP colorIcp;
        // colorIcp.setParameters(true, true);  // Enable downsampling and normal estimation
        colorIcp.setParameters(
            true,   // downsample
            true,   // estimate normals
            0.01,   // voxel resolution
            0.05,   // normal estimation radius
            0.04,   // search radius
            0.968,  // color ICP lambda
            0.05    // max correspondence distance for ICP
        );
        try {
            // Perform the registration
            Eigen::Matrix4d transformation = colorIcp.perform(movingCloud, originCloud);
            
            // Convert Eigen transformation matrix to OpenCV format
            cv::Mat transform(4, 4, CV_32F);
            for(int i = 0; i < 4; i++)
                for(int j = 0; j < 4; j++)
                    transform.at<float>(i,j) = static_cast<float>(transformation(i,j));

            // Apply transformation to matrix
            clouds[movingCloudIndex].transform_matrix = transform * clouds[movingCloudIndex].transform_matrix;

            // Apply transformation to point cloud data
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
            Eigen::Affine3f transformEigen;
            for(int i = 0; i < 4; i++)
                for(int j = 0; j < 4; j++)
                    transformEigen.matrix()(i,j) = transform.at<float>(i,j);
            pcl::transformPointCloud(*clouds[movingCloudIndex].cloud, *transformedCloud, transformEigen);
            *clouds[movingCloudIndex].cloud = *transformedCloud;

            // Reset DOF adjustments since they're now incorporated into the transform
            clouds[movingCloudIndex].dof_adjustments.fill(0.0f);

            updateCloudTransform(movingCloudIndex);
            updateTransformDisplay();

            QString message = QString("ColorICP alignment completed successfully!");
            QMessageBox::information(this, "Success", message);
        } catch (const std::exception& e) {
            QMessageBox::warning(this, "Warning", QString("ColorICP failed: %1").arg(e.what()));
        }
    }

    void adjustTransform(TransformMode mode, float delta) {
        if (movingCloudIndex < 0 || movingCloudIndex >= clouds.size()) return;
        
        auto& cloudData = clouds[movingCloudIndex];
        int dofIndex = static_cast<int>(mode);
        cloudData.dof_adjustments[dofIndex] += delta;
        
        // Create delta transform matrix for the current adjustment
        cv::Mat deltaTransform = cv::Mat::eye(4, 4, CV_32F);
        
        // Apply translations
        deltaTransform.at<float>(0,3) = delta * (mode == TransformMode::TRANSLATE_X ? 1 : 0);
        deltaTransform.at<float>(1,3) = delta * (mode == TransformMode::TRANSLATE_Y ? 1 : 0);
        deltaTransform.at<float>(2,3) = delta * (mode == TransformMode::TRANSLATE_Z ? 1 : 0);
        
        // Apply rotations
        if (mode == TransformMode::ROTATE_X || mode == TransformMode::ROTATE_Y || mode == TransformMode::ROTATE_Z) {
            cv::Mat rot = cv::Mat::eye(4, 4, CV_32F);
            float angle = delta;
            
            if (mode == TransformMode::ROTATE_X) {
                rot.at<float>(1,1) = cos(angle);
                rot.at<float>(1,2) = -sin(angle);
                rot.at<float>(2,1) = sin(angle);
                rot.at<float>(2,2) = cos(angle);
            }
            else if (mode == TransformMode::ROTATE_Y) {
                rot.at<float>(0,0) = cos(angle);
                rot.at<float>(0,2) = sin(angle);
                rot.at<float>(2,0) = -sin(angle);
                rot.at<float>(2,2) = cos(angle);
            }
            else { // ROTATE_Z
                rot.at<float>(0,0) = cos(angle);
                rot.at<float>(0,1) = -sin(angle);
                rot.at<float>(1,0) = sin(angle);
                rot.at<float>(1,1) = cos(angle);
            }
            
            deltaTransform = deltaTransform * rot;
        }
        
        // Update DOF adjustments for display
        cloudData.dof_adjustments[static_cast<int>(mode)] += delta;
        
        // Apply new transform on top of existing transform
        cloudData.transform_matrix = deltaTransform * cloudData.transform_matrix;
        
        // Transform the actual point cloud data
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        Eigen::Affine3f transformEigen;
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++)
                transformEigen.matrix()(i,j) = deltaTransform.at<float>(i,j);
        pcl::transformPointCloud(*clouds[movingCloudIndex].cloud, *transformedCloud, transformEigen);
        *clouds[movingCloudIndex].cloud = *transformedCloud;
        
        updateCloudTransform(movingCloudIndex);
        updateTransformDisplay();
    }
    
    void updateCloudTransform(int cloudIndex) {
        if (cloudIndex < 0 || cloudIndex >= clouds.size()) return;
        
        auto& cloudData = clouds[cloudIndex];
        
        // Convert transform matrix to VTK format
        vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
        transform->PostMultiply();
        
        vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                matrix->SetElement(i, j, cloudData.transform_matrix.at<float>(i,j));
            }
        }
        
        transform->SetMatrix(matrix);
        cloudData.actor->SetUserTransform(transform);
        renderWindow->Render();
    }
    
    void updateTransformDisplay() {
        if (movingCloudIndex < 0 || originCloudIndex < 0) {
            transformOutput->clear();
            return;
        }
        
        const auto& cloudData = clouds[movingCloudIndex];
        QString output;
        QTextStream stream(&output);
        stream.setRealNumberPrecision(6);
        
        // Show DOF adjustments
        stream << "DOF Adjustments for " << QString::fromStdString(cloudData.cloud_name) << ":\n";
        stream << QString("TX: %1  TY: %2  TZ: %3\n")
                 .arg(cloudData.dof_adjustments[0], 8, 'f', 4)
                 .arg(cloudData.dof_adjustments[1], 8, 'f', 4)
                 .arg(cloudData.dof_adjustments[2], 8, 'f', 4);
        stream << QString("RX: %1  RY: %2  RZ: %3\n\n")
                 .arg(cloudData.dof_adjustments[3], 8, 'f', 4)
                 .arg(cloudData.dof_adjustments[4], 8, 'f', 4)
                 .arg(cloudData.dof_adjustments[5], 8, 'f', 4);
        
        // Show transformation matrix
        stream << "Transform Matrix:\n";
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                stream << QString("%1 ").arg(cloudData.transform_matrix.at<float>(i, j), 8, 'f', 4);
            }
            stream << "\n";
        }
        
        transformOutput->setText(output);
    }

    void updateCloudVisuals() {
        // Update visuals based on selection
        for (size_t i = 0; i < clouds.size(); i++) {
            auto& cloudData = clouds[i];
            if (!cloudData.actor) continue;

            if (i == originCloudIndex) {
                cloudData.actor->GetProperty()->SetPointSize(5);  // Make origin cloud points larger
                cloudData.actor->GetProperty()->SetAmbient(0.5);  // Make it brighter
            } else if (i == movingCloudIndex) {
                cloudData.actor->GetProperty()->SetPointSize(4);  // Make moving cloud points medium size
                cloudData.actor->GetProperty()->SetAmbient(0.3);  // Standard brightness
            } else {
                cloudData.actor->GetProperty()->SetPointSize(2);  // Make other clouds smaller
                cloudData.actor->GetProperty()->SetAmbient(0.1);  // Make them darker
            }
        }
        renderWindow->Render();
    }

    void updateCloudColors() {
        for (size_t i = 0; i < clouds.size(); i++) {
            auto& cloudData = clouds[i];
            if (!cloudData.actor || !cloudData.cloud) continue;

            vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
            colors->SetNumberOfComponents(3);
            colors->SetName("Colors");

            if (showOriginalColors) {
                // Use original RGB values from point cloud
                for (const auto& point : *cloudData.cloud) {
                    colors->InsertNextTuple3(point.r, point.g, point.b);
                }
            } else if (uniqueColorMode) {
                // Generate unique color for this cloud
                float hue = i * 360.0f / clouds.size();
                QColor color = QColor::fromHsv(hue, 255, 255);
                for (size_t j = 0; j < cloudData.cloud->size(); j++) {
                    colors->InsertNextTuple3(color.red(), color.green(), color.blue());
                }
            }

            cloudData.polydata->GetPointData()->SetScalars(colors);
            cloudData.actor->GetProperty()->SetPointSize(3);  // Make points more visible
        }
    }

    void loadPointCloud() {
        QString fileName = QFileDialog::getOpenFileName(this,
            tr("Open Point Cloud"), "", tr("Point Cloud Files (*.pcd)"));
            
        if (fileName.isEmpty()) return;

        CloudData cloudData;
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(fileName.toStdString(), *cloudData.cloud) == -1) {
            QMessageBox::warning(this, "Error", "Failed to load PCD file.");
            return;
        }

        // Generate unique name
        QString baseName = QFileInfo(fileName).baseName();
        QString cloudName = baseName;
        int counter = 1;
        while (std::any_of(clouds.begin(), clouds.end(), 
                        [&](const CloudData& cd) { return cd.cloud_name == cloudName.toStdString(); })) {
            cloudName = baseName + QString("_%1").arg(counter++);
        }
        cloudData.cloud_name = cloudName.toStdString();

        // Create VTK visualization pipeline
        cloudData.polydata = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto vertices = vtkSmartPointer<vtkCellArray>::New();
        auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        
        colors->SetNumberOfComponents(3);
        colors->SetName("Colors");
        
        // Convert PCL cloud to VTK
        points->SetNumberOfPoints(cloudData.cloud->size());
        colors->SetNumberOfTuples(cloudData.cloud->size());
        
        for (size_t i = 0; i < cloudData.cloud->size(); ++i) {
            const auto& p = cloudData.cloud->points[i];
            points->SetPoint(i, p.x, p.y, p.z);
            colors->SetTuple3(i, p.r, p.g, p.b);
            
            vertices->InsertNextCell(1);
            vertices->InsertCellPoint(i);
        }
        
        cloudData.polydata->SetPoints(points);
        cloudData.polydata->SetVerts(vertices);
        cloudData.polydata->GetPointData()->SetScalars(colors);
        
        cloudData.mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        cloudData.mapper->SetInputData(cloudData.polydata);
        
        cloudData.actor = vtkSmartPointer<vtkActor>::New();
        cloudData.actor->SetMapper(cloudData.mapper);
        cloudData.actor->GetProperty()->SetPointSize(2);
        
        // Add to renderer
        renderer->AddActor(cloudData.actor);
        
        // Add to list with selection buttons
        QListWidgetItem* item = new QListWidgetItem(cloudList);
        QCheckBox* checkbox = new QCheckBox(cloudName);
        checkbox->setChecked(true);
        
        connect(checkbox, &QCheckBox::stateChanged, 
                [this, cloudName](int state) {
            auto it = std::find_if(clouds.begin(), clouds.end(),
                [&](const CloudData& cd) { return cd.cloud_name == cloudName.toStdString(); });
            if (it != clouds.end()) {
                it->actor->SetVisibility(state == Qt::Checked);
                renderWindow->Render();
            }
        });
        
        cloudList->setItemWidget(item, checkbox);
        item->setSizeHint(checkbox->sizeHint());
        
        // Add to origin and moving cloud lists
        originCloudList->addItem(cloudName);
        movingCloudList->addItem(cloudName);
        
        // Store cloud data
        clouds.push_back(std::move(cloudData));
        
        // If this is the first cloud, make it the origin cloud
        if (clouds.size() == 1) {
            originCloudIndex = 0;
            originCloudList->setCurrentRow(0);
        }
        
        // Reset camera and render
        renderer->ResetCamera();
        renderWindow->Render();
        
        updateTransformDisplay();
    }

private:
    bool loadPCDFile(const std::string& filename,
                     std::vector<float>& points,
                     std::vector<unsigned char>& colors) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
            return false;
        }

        points.reserve(cloud->points.size() * 3);
        colors.reserve(cloud->points.size() * 3);

        for (const auto& p : cloud->points) {
            points.push_back(p.x);
            points.push_back(p.y);
            points.push_back(p.z);
            colors.push_back(p.r);
            colors.push_back(p.g);
            colors.push_back(p.b);
        }
        return true;
    }

    bool loadPointsFromFile(const std::string& filename, 
                          std::vector<float>& points,
                          std::vector<unsigned char>& colors) {
        // Try loading as PCD first
        if (filename.substr(filename.find_last_of(".") + 1) == "pcd") {
            return loadPCDFile(filename, points, colors);
        }

        // If not PCD or PCD loading failed, try loading as text file
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        bool hasPoints = false;
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            float x, y, z;
            if (ss >> x >> y >> z) {
                points.push_back(x);
                points.push_back(y);
                points.push_back(z);
                hasPoints = true;
                
                // Try to read colors
                float r, g, b;
                if (ss >> r >> g >> b) {
                    // Assuming color values are in [0,1] or [0,255]
                    if (r <= 1.0f && g <= 1.0f && b <= 1.0f) {
                        r *= 255.0f;
                        g *= 255.0f;
                        b *= 255.0f;
                    }
                    colors.push_back(static_cast<unsigned char>(r));
                    colors.push_back(static_cast<unsigned char>(g));
                    colors.push_back(static_cast<unsigned char>(b));
                }
            }
        }
        
        return hasPoints;
    }

    QVTKOpenGLNativeWidget* qvtkWidget;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkOrientationMarkerWidget> orientationWidget;
    QListWidget* cloudList;
    QListWidget* originCloudList;    // List for origin cloud selection
    QListWidget* movingCloudList;    // List for moving cloud selection
    
    // Control panel widgets
    QDoubleSpinBox* stepSizeSpinBox;
    QTextEdit* transformOutput;
    
    // Transform control buttons
    QPushButton *tx_minus_btn, *tx_plus_btn;
    QPushButton *ty_minus_btn, *ty_plus_btn;
    QPushButton *tz_minus_btn, *tz_plus_btn;
    QPushButton *rx_minus_btn, *rx_plus_btn;
    QPushButton *ry_minus_btn, *ry_plus_btn;
    QPushButton *rz_minus_btn, *rz_plus_btn;
    
    // Cloud data and state
    std::vector<CloudData> clouds;
    float stepSize = 0.01f;
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    PointCloudViewer viewer;
    viewer.show();
    return app.exec();
}

#include "qt_manual_alignment_tool.moc"
