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
#include <vtkArrowSource.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkVectorText.h>
#include <vtkFollower.h>

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
#include <iomanip>
#include <iostream>
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
    bool uniqueColorMode = false;      // Changed default to false
    bool showOriginalColors = true;    // Changed default to true

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
        setupUI();
        connectSignals();
    }

private:
    void setupUI() {
        // Create the main widget and layout
        QWidget* centralWidget = new QWidget(this);
        QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
        
        setupVTKWidget();
        setupRightPanel();
        
        mainLayout->addWidget(qvtkWidget, 4);  // Viewer gets 4/5 of the width
        mainLayout->addWidget(scrollArea, 1);   // Right panel gets 1/5 of the width
        
        setCentralWidget(centralWidget);
        resize(1200, 800);
    }
    
    void setupVTKWidget() {
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
    }
    
    void setupRightPanel() {
        // Create right panel with scroll area for controls
        scrollArea = new QScrollArea();
        QWidget* rightPanel = new QWidget();
        QVBoxLayout* rightLayout = new QVBoxLayout(rightPanel);
        
        // Create all sections
        QGroupBox* loadGroup = createLoadSection();
        QGroupBox* selectionGroup = createSelectionSection();
        QGroupBox* transformGroup = createTransformSection();
        QGroupBox* displayGroup = createDisplaySection();
        
        // Add all sections to right panel
        rightLayout->addWidget(loadGroup);
        rightLayout->addWidget(selectionGroup);
        rightLayout->addWidget(transformGroup);
        rightLayout->addWidget(displayGroup);
        rightLayout->addStretch();
        
        scrollArea->setWidget(rightPanel);
        scrollArea->setWidgetResizable(true);
        scrollArea->setMinimumWidth(400);
    }
    
    QGroupBox* createLoadSection() {
        QGroupBox* loadGroup = new QGroupBox("Point Clouds");
        loadGroup->setStyleSheet(
            "QGroupBox { "
            "    background-color: #f8f9fa; "
            "    border: 1px solid #dde1e3; "
            "    border-radius: 6px; "
            "    margin-top: 1ex; "
            "    font-weight: bold; "
            "} "
            "QGroupBox::title { "
            "    color: #2d3436; "
            "}"
        );
        QVBoxLayout* loadLayout = new QVBoxLayout(loadGroup);
        
        QPushButton* loadButton = new QPushButton("Load Point Cloud", loadGroup);
        loadButton->setStyleSheet(
            "QPushButton { "
            "    background-color: #00b894; "
            "    color: white; "
            "    border: none; "
            "    padding: 8px; "
            "    border-radius: 4px; "
            "    font-weight: bold; "
            "} "
            "QPushButton:hover { background-color: #00cec9; }"
            "QPushButton:pressed { background-color: #00a8a3; }"
        );
        connect(loadButton, &QPushButton::clicked, this, &PointCloudViewer::loadPointCloud);
        
        cloudList = new QListWidget(loadGroup);
        cloudList->setStyleSheet(
            "QListWidget { "
            "    background-color: white; "
            "    border: 1px solid #dfe6e9; "
            "    border-radius: 4px; "
            "    padding: 4px; "
            "} "
            "QListWidget::item { "
            "    background-color: #f5f6fa; "
            "    border: 1px solid #dfe6e9; "
            "    border-radius: 4px; "
            "    margin: 2px; "
            "} "
            "QListWidget::item:hover { "
            "    background-color: #dfe6e9; "
            "}"
        );
        cloudList->setSpacing(2);
        
        loadLayout->addWidget(loadButton);
        loadLayout->addWidget(cloudList);
        
        return loadGroup;
    }
    
    QGroupBox* createSelectionSection() {
        QGroupBox* selectionGroup = new QGroupBox("Cloud Selection");
        selectionGroup->setStyleSheet("QGroupBox { font-weight: bold; }");
        QVBoxLayout* selectionLayout = new QVBoxLayout(selectionGroup);
        
        // Origin Cloud Selection with enhanced visuals
        QGroupBox* originGroup = new QGroupBox("Reference/Fixed Cloud");
        originGroup->setStyleSheet(
            "QGroupBox { "
            "    background-color: #f0f8ff; "
            "    border: 2px solid #4a90e2; "
            "    border-radius: 6px; "
            "    margin-top: 1ex; "
            "} "
            "QGroupBox::title { "
            "    color: #4a90e2; "
            "    subcontrol-origin: margin; "
            "    left: 10px; "
            "    padding: 0 5px; "
            "}"
        );
        QVBoxLayout* originLayout = new QVBoxLayout(originGroup);
        
        QLabel* originInstruction = new QLabel("Select the fixed reference cloud that other clouds will align to:");
        originInstruction->setStyleSheet("QLabel { font-style: italic; color: #4a90e2; margin-bottom: 5px; }");
        originInstruction->setWordWrap(true);
        originLayout->addWidget(originInstruction);
        
        originCloudList = new QListWidget(originGroup);
        originCloudList->setStyleSheet(
            "QListWidget { border: 1px solid #4a90e2; border-radius: 4px; }"
            "QListWidget::item:selected { background-color: #4a90e2; color: white; }"
            "QListWidget::item:hover { background-color: #e6f3ff; }"
        );
        originLayout->addWidget(originCloudList);
        
        // Moving Cloud Selection with enhanced visuals
        QGroupBox* movingGroup = new QGroupBox("Moving Cloud");
        movingGroup->setStyleSheet(
            "QGroupBox { "
            "    background-color: #fff5f5; "
            "    border: 2px solid #e74c3c; "
            "    border-radius: 6px; "
            "    margin-top: 1ex; "
            "} "
            "QGroupBox::title { "
            "    color: #e74c3c; "
            "    subcontrol-origin: margin; "
            "    left: 10px; "
            "    padding: 0 5px; "
            "}"
        );
        QVBoxLayout* movingLayout = new QVBoxLayout(movingGroup);
        
        QLabel* movingInstruction = new QLabel("Select the cloud you want to transform and align:");
        movingInstruction->setStyleSheet("QLabel { font-style: italic; color: #e74c3c; margin-bottom: 5px; }");
        movingInstruction->setWordWrap(true);
        movingLayout->addWidget(movingInstruction);
        
        movingCloudList = new QListWidget(movingGroup);
        movingCloudList->setStyleSheet(
            "QListWidget { border: 1px solid #e74c3c; border-radius: 4px; }"
            "QListWidget::item:selected { background-color: #e74c3c; color: white; }"
            "QListWidget::item:hover { background-color: #ffe6e6; }"
        );
        movingLayout->addWidget(movingCloudList);
        
        // Add both selection groups to the selection layout
        selectionLayout->addWidget(originGroup);
        selectionLayout->addWidget(movingGroup);
        
        return selectionGroup;
    }
    
    QGroupBox* createTransformSection() {
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
        
        // ColorICP button with styling
        QPushButton* colorICPButton = new QPushButton("Refine with ColorICP");
        colorICPButton->setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        );
        connect(colorICPButton, &QPushButton::clicked, this, &PointCloudViewer::refineWithColorICP);

        // Visualize TFs button
        QPushButton* visualizeTFsButton = new QPushButton("Visualize Transforms");
        visualizeTFsButton->setStyleSheet(
            "QPushButton { background-color: #e67e22; color: white; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d35400; }"
        );
        connect(visualizeTFsButton, &QPushButton::clicked, this, &PointCloudViewer::visualizeTransforms);

        // Export transforms button
        QPushButton* exportTFsButton = new QPushButton("Export Transforms");
        exportTFsButton->setStyleSheet(
            "QPushButton { background-color: #8e44ad; color: white; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #9b59b6; }"
        );
        connect(exportTFsButton, &QPushButton::clicked, this, &PointCloudViewer::exportTransforms);

        // Reference frame toggle
        QCheckBox* showReferenceFrame = new QCheckBox("Show Reference Frame");
        showReferenceFrame->setChecked(true);
        showReferenceFrame->setStyleSheet("QCheckBox { font-weight: bold; color: #666666; }");
        connect(showReferenceFrame, &QCheckBox::toggled, [this](bool checked) {
            orientationWidget->SetEnabled(checked);
            renderWindow->Render();
        });
        
        // Transform buttons grid
        QGridLayout* transformButtonLayout = createTransformButtons();
        
        // Transform Display
        transformOutput = new QTextEdit();
        transformOutput->setReadOnly(true);
        transformOutput->setMinimumHeight(100);
        
        transformLayout->addLayout(stepLayout);
        transformLayout->addWidget(colorICPButton);
        transformLayout->addWidget(visualizeTFsButton);
        transformLayout->addWidget(exportTFsButton);
        transformLayout->addWidget(showReferenceFrame);
        transformLayout->addLayout(transformButtonLayout);
        transformLayout->addWidget(transformOutput);
        
        return transformGroup;
    }
    
    QGridLayout* createTransformButtons() {
        QGridLayout* transformButtonLayout = new QGridLayout();
        
        // Translation buttons with colors
        tx_minus_btn = new QPushButton("-X");
        tx_plus_btn = new QPushButton("+X");
        ty_minus_btn = new QPushButton("-Y");
        ty_plus_btn = new QPushButton("+Y");
        tz_minus_btn = new QPushButton("-Z");
        tz_plus_btn = new QPushButton("+Z");
        
        // Set colors for translation buttons
        QString xAxisStyle = "QPushButton { background-color: #ff4444; color: white; padding: 6px; border-radius: 4px; } QPushButton:hover { background-color: #cc3333; }";
        QString yAxisStyle = "QPushButton { background-color: #44ff44; color: white; padding: 6px; border-radius: 4px; } QPushButton:hover { background-color: #33cc33; }";
        QString zAxisStyle = "QPushButton { background-color: #4444ff; color: white; padding: 6px; border-radius: 4px; } QPushButton:hover { background-color: #3333cc; }";
        
        tx_minus_btn->setStyleSheet(xAxisStyle);
        tx_plus_btn->setStyleSheet(xAxisStyle);
        ty_minus_btn->setStyleSheet(yAxisStyle);
        ty_plus_btn->setStyleSheet(yAxisStyle);
        tz_minus_btn->setStyleSheet(zAxisStyle);
        tz_plus_btn->setStyleSheet(zAxisStyle);
        
        transformButtonLayout->addWidget(tx_minus_btn, 0, 0);
        transformButtonLayout->addWidget(tx_plus_btn, 0, 1);
        transformButtonLayout->addWidget(ty_minus_btn, 1, 0);
        transformButtonLayout->addWidget(ty_plus_btn, 1, 1);
        transformButtonLayout->addWidget(tz_minus_btn, 2, 0);
        transformButtonLayout->addWidget(tz_plus_btn, 2, 1);
        
        // Rotation buttons with colors
        rx_minus_btn = new QPushButton("-RX");
        rx_plus_btn = new QPushButton("+RX");
        ry_minus_btn = new QPushButton("-RY");
        ry_plus_btn = new QPushButton("+RY");
        rz_minus_btn = new QPushButton("-RZ");
        rz_plus_btn = new QPushButton("+RZ");
        
        // Set colors for rotation buttons
        QString rotationStyle = "QPushButton { background-color: #ff9933; color: white; padding: 6px; border-radius: 4px; } QPushButton:hover { background-color: #cc7a29; }";
        rx_minus_btn->setStyleSheet(rotationStyle);
        rx_plus_btn->setStyleSheet(rotationStyle);
        ry_minus_btn->setStyleSheet(rotationStyle);
        ry_plus_btn->setStyleSheet(rotationStyle);
        rz_minus_btn->setStyleSheet(rotationStyle);
        rz_plus_btn->setStyleSheet(rotationStyle);
        
        transformButtonLayout->addWidget(rx_minus_btn, 0, 2);
        transformButtonLayout->addWidget(rx_plus_btn, 0, 3);
        transformButtonLayout->addWidget(ry_minus_btn, 1, 2);
        transformButtonLayout->addWidget(ry_plus_btn, 1, 3);
        transformButtonLayout->addWidget(rz_minus_btn, 2, 2);
        transformButtonLayout->addWidget(rz_plus_btn, 2, 3);
        
        return transformButtonLayout;
    }
    
    QGroupBox* createDisplaySection() {
        QGroupBox* displayGroup = new QGroupBox("Display Options");
        QVBoxLayout* displayLayout = new QVBoxLayout(displayGroup);
        
        // Create a group box for color mode selection
        QGroupBox* colorModeGroup = new QGroupBox("Point Cloud Colors");
        colorModeGroup->setStyleSheet(
            "QGroupBox { "
            "    background-color: #f8f9fa; "
            "    border: 1px solid #dee2e6; "
            "    border-radius: 4px; "
            "    margin-top: 1ex; "
            "} "
            "QGroupBox::title { "
            "    color: #495057; "
            "    subcontrol-origin: margin; "
            "    left: 7px; "
            "    padding: 0 3px; "
            "}"
        );
        QVBoxLayout* colorModeLayout = new QVBoxLayout(colorModeGroup);
        
        uniqueColorRadio = new QRadioButton("Unique Color per Cloud");
        uniqueColorRadio->setStyleSheet(
            "QRadioButton { color: #2c3e50; padding: 2px; }"
            "QRadioButton:hover { background-color: #e9ecef; border-radius: 3px; }"
        );
        
        originalColorRadio = new QRadioButton("Original Point Colors");
        originalColorRadio->setChecked(true);  // Make original colors default
        originalColorRadio->setStyleSheet(
            "QRadioButton { color: #2c3e50; padding: 2px; }"
            "QRadioButton:hover { background-color: #e9ecef; border-radius: 3px; }"
        );
        
        colorModeLayout->addWidget(originalColorRadio);  // Put original first
        colorModeLayout->addWidget(uniqueColorRadio);
        displayLayout->addWidget(colorModeGroup);
        
        return displayGroup;
    }
    
    void connectSignals() {
        // Connect transform control signals
        connect(stepSizeSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [this](double value) { stepSize = static_cast<float>(value); });
        
        // Translation button connections
        connect(tx_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_X, -stepSize); });
        connect(tx_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_X, stepSize); });
        connect(ty_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_Y, -stepSize); });
        connect(ty_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_Y, stepSize); });
        connect(tz_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_Z, -stepSize); });
        connect(tz_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::TRANSLATE_Z, stepSize); });
        
        // Rotation button connections
        connect(rx_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_X, -stepSize); });
        connect(rx_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_X, stepSize); });
        connect(ry_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_Y, -stepSize); });
        connect(ry_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_Y, stepSize); });
        connect(rz_minus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_Z, -stepSize); });
        connect(rz_plus_btn, &QPushButton::clicked, [this]() { adjustTransform(TransformMode::ROTATE_Z, stepSize); });
        
        // Connect display option signals for color mode
        connect(uniqueColorRadio, &QRadioButton::toggled, [this](bool checked) {
            if (checked) {
                uniqueColorMode = true;
                showOriginalColors = false;
                updateCloudColors();
                qvtkWidget->renderWindow()->Render();
            }
        });
        
        connect(originalColorRadio, &QRadioButton::toggled, [this](bool checked) {
            if (checked) {
                uniqueColorMode = false;
                showOriginalColors = true;
                updateCloudColors();
                qvtkWidget->renderWindow()->Render();
            }
        });

        // Connect cloud selection signals
        connect(originCloudList, &QListWidget::currentRowChanged, [this](int row) {
            originCloudIndex = row;
            updateCloudVisuals();
            updateTransformDisplay();
            
            // Update transform arrows when reference cloud changes
            if (showTransforms) {
                createTransformArrows();
                renderWindow->Render();
            }
        });
        
        connect(movingCloudList, &QListWidget::currentRowChanged, [this](int row) {
            movingCloudIndex = row;
            updateCloudVisuals();
            updateTransformDisplay();
        });
    }

private slots:
    void visualizeTransforms() {
        // Toggle transform visualization
        showTransforms = !showTransforms;
        
        if (showTransforms) {
            createTransformArrows();
        } else {
            clearTransformArrows();
        }
        
        renderWindow->Render();
    }

    void exportTransforms() {
        if (originCloudIndex < 0 || clouds.empty()) {
            QMessageBox::warning(this, "Warning", "Please select a reference cloud first.");
            return;
        }

        // Generate transform report
        QString report = generateTransformReport();
        
        // Output to terminal
        std::cout << "\n=== TRANSFORM EXPORT ===" << std::endl;
        std::cout << report.toStdString() << std::endl;
        std::cout << "========================\n" << std::endl;
        
        // Show in popup dialog for copy/paste
        QDialog* dialog = new QDialog(this);
        dialog->setWindowTitle("Transform Export");
        dialog->setMinimumSize(600, 400);
        
        QVBoxLayout* layout = new QVBoxLayout(dialog);
        
        QLabel* titleLabel = new QLabel("Transformation Matrices and 6DOF Values");
        titleLabel->setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50; margin-bottom: 10px;");
        layout->addWidget(titleLabel);
        
        QTextEdit* textEdit = new QTextEdit(dialog);
        textEdit->setPlainText(report);
        textEdit->setFont(QFont("Courier", 10)); // Monospace font for better alignment
        textEdit->selectAll(); // Select all text for easy copying
        layout->addWidget(textEdit);
        
        QHBoxLayout* buttonLayout = new QHBoxLayout();
        
        QPushButton* copyButton = new QPushButton("Copy All", dialog);
        copyButton->setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2980b9; }"
        );
        connect(copyButton, &QPushButton::clicked, [textEdit]() {
            textEdit->selectAll();
            textEdit->copy();
        });
        
        QPushButton* closeButton = new QPushButton("Close", dialog);
        closeButton->setStyleSheet(
            "QPushButton { background-color: #95a5a6; color: white; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #7f8c8d; }"
        );
        connect(closeButton, &QPushButton::clicked, dialog, &QDialog::accept);
        
        buttonLayout->addWidget(copyButton);
        buttonLayout->addStretch();
        buttonLayout->addWidget(closeButton);
        layout->addLayout(buttonLayout);
        
        dialog->exec();
        dialog->deleteLater();
    }

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
        
        // Update transform arrows if they are visible
        if (showTransforms) {
            createTransformArrows();
        }
        
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
        
        // Create widget for cloud control
        QWidget* cloudControlWidget = new QWidget;
        QHBoxLayout* controlLayout = new QHBoxLayout(cloudControlWidget);
        controlLayout->setContentsMargins(2, 2, 2, 2);
        controlLayout->setSpacing(4);

        // Visibility toggle button
        QPushButton* visibilityButton = new QPushButton;
        visibilityButton->setFixedSize(24, 24);
        visibilityButton->setToolTip("Toggle cloud visibility");
        visibilityButton->setCursor(Qt::PointingHandCursor);
        visibilityButton->setText("👁");
        visibilityButton->setStyleSheet(
            "QPushButton { "
            "    background-color: #4a90e2; "
            "    border: none; "
            "    border-radius: 12px; "
            "    color: white; "
            "    font-size: 14px; "
            "} "
            "QPushButton:hover { background-color: #357abd; }"
            "QPushButton:pressed { background-color: #2d6da3; }"
        );

        // Cloud name label with color indicator
        QLabel* nameLabel = new QLabel(cloudName);
        nameLabel->setStyleSheet(QString("QLabel { color: #2c3e50; padding: 2px; }"));
        
        // Remove button
        QPushButton* removeButton = new QPushButton;
        removeButton->setFixedSize(20, 20);
        removeButton->setToolTip("Remove cloud");
        removeButton->setCursor(Qt::PointingHandCursor);
        removeButton->setStyleSheet(
            "QPushButton { "
            "    background-color: #ff4757; "
            "    border: none; "
            "    border-radius: 10px; "
            "    color: white; "
            "    font-weight: bold; "
            "} "
            "QPushButton:hover { background-color: #ff6b81; }"
            "QPushButton:pressed { background-color: #ee5253; }"
        );
        removeButton->setText("×");

        controlLayout->addWidget(visibilityButton);
        controlLayout->addWidget(nameLabel, 1);
        controlLayout->addWidget(removeButton);

        QListWidgetItem* item = new QListWidgetItem(cloudList);
        cloudList->setItemWidget(item, cloudControlWidget);
        item->setSizeHint(cloudControlWidget->sizeHint());
        
        // Connect visibility toggle
        connect(visibilityButton, &QPushButton::clicked, [this, visibilityButton, cloudName]() {
            auto it = std::find_if(clouds.begin(), clouds.end(),
                [&](const CloudData& cd) { return cd.cloud_name == cloudName.toStdString(); });
            if (it != clouds.end()) {
                bool isVisible = it->actor->GetVisibility();
                it->actor->SetVisibility(!isVisible);
                visibilityButton->setStyleSheet(
                    QString("QPushButton { "
                    "    background-color: %1; "
                    "    border: none; "
                    "    border-radius: 12px; "
                    "    color: white; "
                    "    font-size: 14px; "
                    "} "
                    "QPushButton:hover { background-color: %2; }"
                    "QPushButton:pressed { background-color: %3; }")
                    .arg(!isVisible ? "#4a90e2" : "#95a5a6")
                    .arg(!isVisible ? "#357abd" : "#7f8c8d")
                    .arg(!isVisible ? "#2d6da3" : "#666e6f")
                );
                renderWindow->Render();
            }
        });

        // Connect remove button
        connect(removeButton, &QPushButton::clicked, [this, cloudName]() {
            // Find the cloud index
            auto it = std::find_if(clouds.begin(), clouds.end(),
                [&](const CloudData& cd) { return cd.cloud_name == cloudName.toStdString(); });
            
            if (it != clouds.end()) {
                int cloudIndex = std::distance(clouds.begin(), it);
                
                // Update selection indices before removal
                if (cloudIndex == originCloudIndex) {
                    originCloudIndex = -1;  // Clear origin selection
                }
                if (cloudIndex == movingCloudIndex) {
                    movingCloudIndex = -1;  // Clear moving selection
                }
                
                // Adjust indices for clouds after the removed one
                if (originCloudIndex > cloudIndex) {
                    originCloudIndex--;
                }
                if (movingCloudIndex > cloudIndex) {
                    movingCloudIndex--;
                }

                // Remove from renderer first
                renderer->RemoveActor(it->actor);
                
                // First, remove from cloud list widget
                for(int i = cloudList->count() - 1; i >= 0; --i) {
                    QWidget* widget = cloudList->itemWidget(cloudList->item(i));
                    if (widget) {
                        QLabel* label = widget->findChild<QLabel*>();
                        if (label && label->text() == cloudName) {
                            delete cloudList->takeItem(i);
                            break;
                        }
                    }
                }

                // Then, safely remove from origin list widget
                int originCurrentRow = originCloudList->currentRow();
                for(int i = originCloudList->count() - 1; i >= 0; --i) {
                    QListWidgetItem* item = originCloudList->item(i);
                    if (item && item->text() == cloudName) {
                        delete originCloudList->takeItem(i);
                        if (i == originCurrentRow) {
                            if (originCloudList->count() > 0) {
                                originCloudList->setCurrentRow(std::min(i, originCloudList->count() - 1));
                            }
                        }
                        break;
                    }
                }
                
                // Finally, safely remove from moving list widget
                int movingCurrentRow = movingCloudList->currentRow();
                for(int i = movingCloudList->count() - 1; i >= 0; --i) {
                    QListWidgetItem* item = movingCloudList->item(i);
                    if (item && item->text() == cloudName) {
                        delete movingCloudList->takeItem(i);
                        if (i == movingCurrentRow) {
                            if (movingCloudList->count() > 0) {
                                movingCloudList->setCurrentRow(std::min(i, movingCloudList->count() - 1));
                            }
                        }
                        break;
                    }
                }

                // Remove from clouds vector last
                clouds.erase(it);
                
                // Clear transform display if no clouds are selected
                if (originCloudIndex == -1 || movingCloudIndex == -1) {
                    transformOutput->clear();
                }
                
                // Update visualization
                updateCloudVisuals();
                
                // Update transform arrows if they are visible
                if (showTransforms) {
                    createTransformArrows();
                }
                
                renderWindow->Render();
            }
        });
        
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

    void createTransformArrows() {
        clearTransformArrows();
        
        if (originCloudIndex < 0 || clouds.empty()) return;
        
        // Get origin cloud position as reference
        cv::Mat originTransform = clouds[originCloudIndex].transform_matrix;
        cv::Point3f originPos(originTransform.at<float>(0,3), 
                             originTransform.at<float>(1,3), 
                             originTransform.at<float>(2,3));
        
        for (size_t i = 0; i < clouds.size(); ++i) {
            if (i == originCloudIndex) continue; // Skip origin cloud
            
            auto& cloudData = clouds[i];
            cv::Mat transform = cloudData.transform_matrix;
            
            // Extract target cloud position
            cv::Point3f targetPos(transform.at<float>(0,3), 
                                 transform.at<float>(1,3), 
                                 transform.at<float>(2,3));
            
            // Calculate vector from origin to target
            cv::Point3f direction = targetPos - originPos;
            float distance = cv::norm(direction);
            
            if (distance < 1e-6) continue; // Skip if clouds are at same position
            
            // Normalize direction vector
            direction = direction * (1.0f / distance);
            
            // Create arrow source with proportional dimensions
            vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New();
            arrowSource->SetTipLength(0.15);  // Tip is 15% of arrow length
            arrowSource->SetTipRadius(0.08);  // Increased tip radius for thicker appearance
            arrowSource->SetShaftRadius(0.05); // Increased shaft radius for thicker appearance
            
            // Create transform for arrow
            vtkSmartPointer<vtkTransform> arrowTransform = vtkSmartPointer<vtkTransform>::New();
            
            // Position arrow at origin cloud position
            arrowTransform->Translate(originPos.x, originPos.y, originPos.z);
            
            // Orient arrow to point toward target cloud
            // Calculate rotation to align arrow with direction vector
            cv::Point3f defaultDirection(1.0f, 0.0f, 0.0f); // Arrow default points along X-axis
            cv::Point3f rotationAxis = defaultDirection.cross(direction);
            float rotationAngle = acos(std::max(-1.0f, std::min(1.0f, defaultDirection.dot(direction))));
            
            if (cv::norm(rotationAxis) > 1e-6) {
                // Normalize rotation axis
                rotationAxis = rotationAxis * (1.0f / cv::norm(rotationAxis));
                
                // Convert to degrees and apply rotation
                arrowTransform->RotateWXYZ(rotationAngle * 180.0 / M_PI, 
                                          rotationAxis.x, rotationAxis.y, rotationAxis.z);
            }
            
            // Scale arrow to match the actual distance (dynamic length, no limits)
            arrowTransform->Scale(distance, 0.05, 0.05); // Length = distance, increased fixed width/height for thicker arrows
            
            // Apply transform to arrow
            vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = 
                vtkSmartPointer<vtkTransformPolyDataFilter>::New();
            transformFilter->SetInputConnection(arrowSource->GetOutputPort());
            transformFilter->SetTransform(arrowTransform);
            
            // Create mapper
            vtkSmartPointer<vtkPolyDataMapper> arrowMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            arrowMapper->SetInputConnection(transformFilter->GetOutputPort());
            
            // Create actor
            vtkSmartPointer<vtkActor> arrowActor = vtkSmartPointer<vtkActor>::New();
            arrowActor->SetMapper(arrowMapper);
            
            // Set arrows to gray color for consistency
            arrowActor->GetProperty()->SetColor(0.6, 0.6, 0.6); // Gray color
            arrowActor->GetProperty()->SetOpacity(0.9);
            arrowActor->GetProperty()->SetLineWidth(4.0); // Increased line width
            
            // Add arrow to renderer
            renderer->AddActor(arrowActor);
            transformArrows.push_back(arrowActor);
            
            // Create distance label at midpoint of arrow with better visibility
            cv::Point3f labelPos = originPos + direction * (distance * 0.6f); // Position at 60% along arrow
            
            std::ostringstream labelText;
            labelText << cloudData.cloud_name << "\n" << std::fixed << std::setprecision(2) << distance << "m";
            
            vtkSmartPointer<vtkVectorText> textSource = vtkSmartPointer<vtkVectorText>::New();
            textSource->SetText(labelText.str().c_str());
            
            vtkSmartPointer<vtkPolyDataMapper> textMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            textMapper->SetInputConnection(textSource->GetOutputPort());
            
            vtkSmartPointer<vtkFollower> textActor = vtkSmartPointer<vtkFollower>::New();
            textActor->SetMapper(textMapper);
            textActor->SetPosition(labelPos.x, labelPos.y, labelPos.z + 0.05); // Fixed offset
            
            // Fixed text scale for consistent visibility
            textActor->SetScale(0.05, 0.05, 0.05);
            
            // High contrast text with outline effect
            textActor->GetProperty()->SetColor(1.0, 1.0, 1.0); // White text
            textActor->GetProperty()->SetAmbient(1.0);
            textActor->GetProperty()->SetDiffuse(0.0);
            textActor->GetProperty()->SetSpecular(0.0);
            textActor->SetCamera(renderer->GetActiveCamera());
            
            // Add label to renderer
            renderer->AddActor(textActor);
            transformLabels.push_back(textActor);
        }
    }

    void clearTransformArrows() {
        // Remove all transform arrows and labels from renderer
        for (auto& arrow : transformArrows) {
            renderer->RemoveActor(arrow);
        }
        for (auto& label : transformLabels) {
            renderer->RemoveActor(label);
        }
        
        transformArrows.clear();
        transformLabels.clear();
    }

    QString generateTransformReport() {
        QString report;
        QTextStream stream(&report);
        stream.setRealNumberPrecision(6);
        
        const auto& originCloud = clouds[originCloudIndex];
        stream << "Reference Cloud: " << QString::fromStdString(originCloud.cloud_name) << "\n";
        stream << "Total Clouds: " << clouds.size() << "\n\n";
        
        // Get origin cloud transform as reference
        cv::Mat originTransform = originCloud.transform_matrix;
        cv::Mat originInverse;
        cv::invert(originTransform, originInverse);
        
        for (size_t i = 0; i < clouds.size(); ++i) {
            const auto& cloudData = clouds[i];
            
            stream << "Cloud: " << QString::fromStdString(cloudData.cloud_name);
            if (i == originCloudIndex) {
                stream << " (REFERENCE)\n";
                stream << "Extrinsic Matrix (Identity):\n";
                stream << "1.000000  0.000000  0.000000  0.000000\n";
                stream << "0.000000  1.000000  0.000000  0.000000\n";
                stream << "0.000000  0.000000  1.000000  0.000000\n";
                stream << "0.000000  0.000000  0.000000  1.000000\n";
                stream << "6DOF: TX=0.000000, TY=0.000000, TZ=0.000000, RX=0.000000, RY=0.000000, RZ=0.000000\n\n";
                continue;
            }
            
            stream << "\n";
            
            // Calculate relative transform from current cloud to origin cloud
            cv::Mat relativeTransform = originInverse * cloudData.transform_matrix;
            
            // Extract translation
            float tx = relativeTransform.at<float>(0, 3);
            float ty = relativeTransform.at<float>(1, 3);
            float tz = relativeTransform.at<float>(2, 3);
            
            // Extract rotation matrix and convert to Euler angles (XYZ order)
            cv::Mat rotMat = relativeTransform(cv::Rect(0, 0, 3, 3));
            float rx, ry, rz;
            
            // Extract Euler angles from rotation matrix (ZYX convention)
            ry = asin(-rotMat.at<float>(2, 0));
            if (cos(ry) > 1e-6) {
                rx = atan2(rotMat.at<float>(2, 1), rotMat.at<float>(2, 2));
                rz = atan2(rotMat.at<float>(1, 0), rotMat.at<float>(0, 0));
            } else {
                rx = atan2(-rotMat.at<float>(1, 2), rotMat.at<float>(1, 1));
                rz = 0;
            }
            
            // Output extrinsic matrix
            stream << "Extrinsic Matrix:\n";
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    stream << QString("%1  ").arg(relativeTransform.at<float>(row, col), 8, 'f', 6);
                }
                stream << "\n";
            }
            
            // Output 6DOF values
            stream << QString("6DOF: TX=%1, TY=%2, TZ=%3, RX=%4, RY=%5, RZ=%6\n")
                     .arg(tx, 8, 'f', 6)
                     .arg(ty, 8, 'f', 6)
                     .arg(tz, 8, 'f', 6)
                     .arg(rx, 8, 'f', 6)
                     .arg(ry, 8, 'f', 6)
                     .arg(rz, 8, 'f', 6);
            
            // Calculate distance from origin
            float distance = sqrt(tx*tx + ty*ty + tz*tz);
            stream << QString("Distance from reference: %1 meters\n\n").arg(distance, 0, 'f', 3);
        }
        
        return report;
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
    
    // UI widgets
    QScrollArea* scrollArea;
    QListWidget* cloudList;
    QListWidget* originCloudList;    // List for origin cloud selection
    QListWidget* movingCloudList;    // List for moving cloud selection
    
    // Control panel widgets
    QDoubleSpinBox* stepSizeSpinBox;
    QTextEdit* transformOutput;
    
    // Radio buttons for color mode
    QRadioButton* uniqueColorRadio;
    QRadioButton* originalColorRadio;
    
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
    
    // Transform visualization
    std::vector<vtkSmartPointer<vtkActor>> transformArrows;
    std::vector<vtkSmartPointer<vtkFollower>> transformLabels;
    bool showTransforms = false;
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    PointCloudViewer viewer;
    viewer.show();
    return app.exec();
}

#include "qt_manual_alignment_tool.moc"
