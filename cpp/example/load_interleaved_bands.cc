#include <iostream>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Arrays/Slicer.h>
#include <casacore/casa/Arrays/Cube.h>
#include <casacore/ms/MSSel/MSSelection.h>
#include <casacore/ms/MSSel/MSSelectionTools.h> // for mssSetData
#include <casacore/casa/Containers/Record.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSMainColumns.h>
#include <casacore/ms/MeasurementSets/MSAntennaColumns.h>
#include <Eigen/Core>
#include <casacore/ms/MeasurementSets/MSFieldColumns.h>
#include <casacore/ms/MSOper/MSMetaData.h>   // for nRows
#include <casacore/tables/TaQL/ExprNode.h>   // for selection
#include <casacore/ms/MSSel/MSSelector.h>    // for MSSelector
#include <casacore/tables/TaQL/TableParse.h> // for TAQL expression
#include <casacore/ms/MSOper/MSKeys.h>
#include <casarest/msvis/MSVis/SubMS.h> // path not up-to-date inside... installation-related?

// Row partition works well, issues with in-row partition at the moment (chans, corr)

int main(int nargs, char const **args) {

    // from example at: http://casacore.github.io/casacore/classcasacore_1_1MSSelection.html

    string msName = "../ms/CYG-C-6680-64.MS";
    // Create full MeasurementSet
    ::casacore::MeasurementSet ms(msName); 
    // General info on the full ms
    ::casacore::MSMetaData meta_data_ms(&ms, 1.); 
    auto nChannels_ms = meta_data_ms.nChans();
    int nScans_ms = meta_data_ms.nScans();
    // Check spectral content of the full MS
    ::casacore::ArrayColumn<::casacore::Double> freqCols_ms(ms.spectralWindow(), ::casacore::MSSpectralWindow::columnName(::casacore::MSSpectralWindow::CHAN_FREQ));
	Eigen::VectorXd frequencies_ms = Eigen::VectorXd::Map(freqCols_ms(0).data(), freqCols_ms(0).nelements(), 1);

    // // Setup any sub-expressiosn of interest directly (probable issue at this stage)
    // ::casacore::MSSelection select;
    // select.setSpwExpr("0:0;1;2"); // "0:0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20" ("0:0~63^2"); // problem here: why?? (does not seems to work...)
    // select.setScanExpr("0~10"); // works as intended
    // // ::casacore::TableExprNode node = select.toTableExprNode(&ms);
    // //::casacore::MSSelection select(&ms, ::casacore::MSSelection::MSSMode::PARSE_NOW, "", "", "", "0:0~63^2", "", "", "", "", "", "", "", "");
    // ::casacore::TableExprNode node = select.toTableExprNode(&ms);
    // // ::casacore::MSSelection::MSSelection(&ms, MSSMode, timeExpr, antennaExpr, fieldExpr, spwExpr,
    // //udDistExpr, taqlExpr, polmExpr, scanExpr, arrayExpr, stateExpr, obsExpr, feedExpr);

    // // Create a table and a MS representing the selection
    // ::casacore::Table tablesel(ms.tableName(), ::casacore::Table::Update);
    // ::casacore::MeasurementSet mssel(tablesel(node, node.nrow()));


    // // OPTION 2
    // //
    // // Fill in the expression in the various strings that are passed for
    // // parsing to the MSSelection object later.
    // //
    // ::casacore::String fieldStr,timeStr,spwStr,baselineStr,
    // uvdistStr,taqlStr,scanStr,arrayStr, polnStr,stateObsModeStr,
    // observationStr;
    // // ::casacore::String baselineStr="1&2";
    // // ::casacore::String timeStr="*+0:10:0";
    // // ::casacore::String fieldStr="CygA*";
    // spwStr = "0:0;1;2;3"; // does not seem to work as intended!
    // scanStr = "0~10";
    // //
    // // Instantiate the MS and the MSInterface objects.
    // //
    // ::casacore::MS mssel(ms);
    // ::casacore::MSInterface msInterface(ms);
    // //
    // // Setup the MSSelection thingi
    // //
    // ::casacore::MSSelection select;
    // select.reset(msInterface,::casacore::MSSelection::PARSE_NOW,
    // timeStr,baselineStr,fieldStr,spwStr,
    // uvdistStr,taqlStr,polnStr,scanStr,arrayStr,
    // stateObsModeStr,observationStr);
    // if (select.getSelectedMS(mssel, "../ms/subms.MS"))
    // std::cerr << "Got the selected MS!" << endl;
    // else
    // std::cerr << "The set of expressions resulted into null-selection";
    
    // // Check spectral content
    // auto selected_spw = msInterface.spectralWindow();
    // ::casacore::ArrayColumn<::casacore::Double> freqC(msInterface.spectralWindow(), ::casacore::MSSpectralWindow::columnName(::casacore::MSSpectralWindow::CHAN_FREQ));
	// Eigen::VectorXd freq = Eigen::VectorXd::Map(freqC(0).data(), freqC(0).nelements(), 1); // stil 64 entries: why?


    // // With TAQL expression
    // ::casacore::Table seltab = ::casacore::tableCommand("select from ../ms/CYG-C-6680-64.MS where ANTENNA1==0 && ANTENNA2==1");
    // std::cout << seltab.nrow() << std::endl;

    // // Getting final sub-MS file (+ saving on the hard drive)
    // auto channels = select.getChanList(&mssel);
    // std::cout << channels.nrow() << std::endl;
    // ::casacore::Vector<::casacore::Vector<::casacore::Slice>> chanSlices;
    // select.getChanSlices(chanSlices, &mssel);
    // std::cout << chanSlices[0].nelements() << std::endl;
    // ::casacore::Vector<::casacore::Vector<::casacore::Slice>> corrSlices; // problem with corrSlices, or sth else?
    // select.getCorrSlices(corrSlices, &mssel);
    // std::cout << corrSlices[0].nelements() << std::endl;
    // auto spwList = select.getSpwList(&mssel);
    // auto spwDDIDList = select.getSPWDDIDList(&mssel);
    // std::cout << spwList.nelements() << std::endl;
    // std::cout << spwDDIDList.nelements() << std::endl;
    // ::casacore::Bool test = select.getSelectedMS(mssel, "../ms/subms.MS");

    // // OPTION 3
    // ::casacore::Vector<::casacore::Vector<::casacore::Slice> > chanSlices;
    // ::casacore::Vector<::casacore::Vector<::casacore::Slice> > corrSlices;
    // ::casacore::MS mssel;
    // ::casacore::MSSelection select;

    // ::casacore::Bool test = ::casacore::mssSetData(ms, mssel, chanSlices, corrSlices, "../ms/subms.MS",
    //  "", "", "", "0:0", // the syntax spanning mutliple frequencies does not seem to work...
    //  "", "", "", "0~10", "", "", "", 1, &select); // size issue somwhere here... the spectral selection does not seem active, at least in the final MS. Issue in the documentation?
    // std::cout << chanSlices[0].nelements() << std::endl; // shoudl be equal to the number of selected channels
    // std::cout << corrSlices[0].nelements() << std::endl;
    

    // OPTION 4
    ::casacore::MSInterface msLike(ms);
    ::casacore::MSSelection select;

    ::casacore::String baselineStr= ""; //"1&2";
    ::casacore::String timeStr= "";  //"*+0:10:0";
    ::casacore::String fieldStr= ""; //"CygA*";
    ::casacore::String spwStr = "0:0;1;2;3"; // does not seem to work as intended!
    ::casacore::String scanStr = ""; //"0~10"
    ::casacore::String polnStr = "RR,LL,LR,RL";
    ::casacore::String arrayStr = "";
    ::casacore::String taqlStr = "";
    ::casacore::String uvdistStr = "";
    ::casacore::String stateObsModeStr = "";
    ::casacore::String observationStr = "";
    select.reset(msLike,::casacore::MSSelection::PARSE_NOW,
    timeStr,baselineStr,fieldStr,spwStr,
    uvdistStr,taqlStr,polnStr,scanStr,arrayStr,
    stateObsModeStr,observationStr);

    ::casacore::MS mssel;
    ::casacore::Bool test = select.getSelectedMS(mssel, "../ms/subms.MS"); // does not perform the channel selection as originally expected


    // option 5 (with casarest) [comiplation issue at the moment]
    // casa::SubMS subms(ms);
    // ::casacore::String spwStr2 = "0:0;1;2;3";
    // ::casacore::Vector<::casacore::Int> steps(1);
    // steps[0] = 1;
    // ::casacore::Bool test_subms = subms.selectSpw(spwStr2, steps);
    // ::casacore::Bool test_subms = subms.makeSubMS("../ms/subms.MS", "DATA"); // this should hopefully split the MS as expected: to be confirmed

    // // Check content of the final sub-MS (mssel)
    // ::casacore::MSMetaData meta_data(&mssel, 1.);
    // int nScans = meta_data.nScans();
    // auto nChannels = meta_data.nChans(); // table is empty?? why?
    // ::casacore::ArrayColumn<::casacore::Double> freqCols(mssel.spectralWindow(), ::casacore::MSSpectralWindow::columnName(::casacore::MSSpectralWindow::CHAN_FREQ));
	// Eigen::VectorXd frequencies = Eigen::VectorXd::Map(freqCols(0).data(), freqCols(0).nelements(), 1);

    return 0;
}