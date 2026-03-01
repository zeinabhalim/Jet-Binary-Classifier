#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/contrib/Nsubjettiness.hh"
#include <fstream>
#include <vector>
#include <cmath>



using namespace Pythia8;
using namespace fastjet;
using namespace fastjet::contrib;

int main() {

    Pythia pythia;

    // Proton-proton at 13 TeV
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");     // Hard QCD processes
    pythia.readString("PhaseSpace:pTHatMin = 50."); 
    pythia.init();
    
    int nEvents = 5000;
    
    //Clusters particles into cone-like jets with Radius R = 0.4
    JetDefinition jet_def(antikt_algorithm, 0.4);

      std::ofstream outfile("jets_pythia.csv");
        if (!outfile.is_open()) {
          std::cerr << "ERROR: Could not open jets_pythia.csv for writing\n";
          return 1;
            }
outfile << "mass,multiplicity,pT,tau21,tau32,girth,label\n";

   
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (!pythia.next()) continue;
        
      //Converts Pythia particles into FastJet objects
        std::vector<PseudoJet> particles;

        for (int i = 0; i < pythia.event.size(); ++i) {
        
        //to include the intermidate particles and invisible particles
         if (!pythia.event[i].isFinal()) continue;
    if (!pythia.event[i].isVisible()) continue;

    PseudoJet pj(
        pythia.event[i].px(),
        pythia.event[i].py(),
        pythia.event[i].pz(),
        pythia.event[i].e()
    );
    pj.set_user_index(i); 
    particles.push_back(pj);
}
        ClusterSequence cs(particles, jet_def);
        std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(20.0));
        if (jets.size() > 2) jets.resize(2);  // only 2 hardest jets

     // Labeling gluons and quark jets
          for (auto &jet : jets) {

    int label = -1;
    double minDR = 999.0;

    for (int i = 0; i < pythia.event.size(); ++i) {
        if (!pythia.event[i].isParton()) continue;

        PseudoJet parton(
            pythia.event[i].px(),
            pythia.event[i].py(),
            pythia.event[i].pz(),
            pythia.event[i].e()
        );

        double dR = jet.delta_R(parton);

        if (dR < 0.3 && dR < minDR) {
            minDR = dR;
            int id = abs(pythia.event[i].id());
            if (id <= 6) label = 0;
            else if (id == 21) label = 1;
        }
    }

    if (label == -1) continue;


//ESTIMATING THE JET STURCTURE FEATURES

// mass observable
double mass = jet.m();   // define mass
double jet_pt_total = 0.0; // sum of constituent pt for girth normalization

// Count charged and neutral constituents
int n_charged = 0;
int n_neutral = 0;
for (auto &p : jet.constituents()) {
    if (p.user_index() < 0) continue; // safety
    jet_pt_total += p.pt();          // accumulate pt
    if (pythia.event[p.user_index()].isCharged()) n_charged++;
    else n_neutral++;
}

// Compute jet eccentricity
double sum_px2 = 0.0, sum_py2 = 0.0, sum_pxpy = 0.0;
for (auto &p : jet.constituents()) {
    double px = p.px() - jet.px()/jet.constituents().size();
    double py = p.py() - jet.py()/jet.constituents().size();
    sum_px2 += px*px;
    sum_py2 += py*py;
    sum_pxpy += px*py;
}
double lambda1 = 0.5*(sum_px2 + sum_py2 + sqrt((sum_px2 - sum_py2)*(sum_px2 - sum_py2) + 4*sum_pxpy*sum_pxpy));
double lambda2 = 0.5*(sum_px2 + sum_py2 - sqrt((sum_px2 - sum_py2)*(sum_px2 - sum_py2) + 4*sum_pxpy*sum_pxpy));
double eccentricity = (lambda1 > 0) ? 1.0 - lambda2/lambda1 : 0.0;

// Girth (jet width)
double girth = 0.0;
for (auto &p : jet.constituents()) {
    girth += p.pt() * jet.delta_R(p);
}
if (jet_pt_total > 0) girth /= jet_pt_total; // normalize by total jet pt

// N-subjettiness
Nsubjettiness tau1(1, OnePass_KT_Axes(), UnnormalizedMeasure(1.0));
Nsubjettiness tau2(2, OnePass_KT_Axes(), UnnormalizedMeasure(1.0));
Nsubjettiness tau3(3, OnePass_KT_Axes(), UnnormalizedMeasure(1.0));

double t1 = tau1(jet);
double t2 = tau2(jet);
double t3 = tau3(jet);

double tau21 = (t1 > 0) ? t2/t1 : 0;
double tau32 = (t2 > 0) ? t3/t2 : 0;

// ---------------- OUTPUT ----------------
outfile << mass << ","
        << n_charged << ","
        << n_neutral << ","
        << tau21 << ","
        << tau32 << ","
        << girth << ","
        << eccentricity << ","
        << label << "\n";


           }
    }

    outfile.close();
    pythia.stat();
    return 0;
}

