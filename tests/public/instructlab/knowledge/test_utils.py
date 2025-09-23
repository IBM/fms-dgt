# Local
from fms_dgt.public.databuilders.instructlab.knowledge.utils import parse_qa_v2


def test_qa_parser():
    text = """[Question]
    [Question]
    What is photosynthesis and why is it considered one of the most important biological processes on Earth?
    [Answer]
    Photosynthesis is the process by which plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose, or sugar. During this process, carbon dioxide and water are absorbed, and oxygen is released as a byproduct. Photosynthesis is considered one of the most important biological processes because it liberates oxygen and consumes carbon dioxide, making the Earth's atmosphere hospitable. It also directly or indirectly fulfills all of our food requirements and many of our needs for fiber and building materials. Additionally, the energy stored in fossil fuels like petroleum, natural gas, and coal originally came from the sun via photosynthesis.
    [End]
    
    [Question]
    How do plants obtain and use energy from sunlight during photosynthesis?
    [Answer]
    Plants obtain energy from sunlight through photosynthesis, which occurs in photosynthetic reaction centers located in leaves. These tiny units convert light energy into chemical energy, the form used by all living organisms. During photosynthesis, plants use the energy from sunlight to convert carbon dioxide from the air into sugars, starches, and other high-energy carbohydrates, while releasing oxygen. This stored chemical energy is later used by the plant and other organisms that consume the plant for their energy needs.
    [End]
    
    [Question]
    Multiple Choice: Which of the following is NOT a direct result of photosynthesis?
    Options:
    a) Production of oxygen
    b) Conversion of carbon dioxide into sugars
    c) Release of heat energy
    d) Storage of energy in chemical form\n[Answer]
    c) Release of heat energy
    [End]
    
    [Question]
    Explain how the extinction of the dinosaurs might be linked to photosynthesis.
    [Answer]
    One theory suggesting the extinction of the dinosaurs proposes that a catastrophic event, such as a comet, meteor, or volcanic eruption, ejected a large amount of material into the atmosphere, severely reducing the amount of sunlight reaching the Earth. This reduction in sunlight would have significantly impacted photosynthesis, leading to the death of many plants. Consequently, the herbivorous dinosaurs that depended on these plants for food would have starved, and the carnivorous dinosaurs that preyed on the herbivores would have also been affected, leading to a mass extinction.
    [End]
    
    [Question]
    How is cellulose, a product of photosynthesis, important in both energy production and as a material?
    [Answer]
    Cellulose, a carbohydrate produced through photosynthesis, makes up the bulk of dry wood and other plant material. It is important in energy production because it can be burned to release stored energy as heat, converting it back into carbon dioxide. This process is used in many parts of the world where firewood is a critical fuel source. Additionally, cellulose can be converted into sugars and subsequently into ethanol, which can be used as a biofuel. As a material, wood is used for building and other purposes, and cellulose is also used to make paper and textiles.
    [End]

    [Question]
    True or False: Most of the energy we use today, such as coal, oil, and natural gas, originally came from solar energy through photosynthesis.
    [Answer]
    True.
    [End]
    
    [Question]
    Essay Question: Discuss the significance of understanding photosynthesis in the context of global climate change and future energy needs.
    [Answer]
    Understanding photosynthesis is crucial in the context of global climate change and future energy needs for several reasons. Firstly, photosynthesis plays a key role in regulating the Earth's atmosphere by absorbing carbon dioxide, a major greenhouse gas. Therefore, comprehending the mechanisms of photosynthesis can help in developing strategies to mitigate climate change, such as enhancing carbon sequestration through reforestation and improved agricultural practices. Secondly, photosynthesis research can contribute to the development of more efficient crops, which is essential for ensuring food security in a changing climate. Additionally, insights into the energy-harvesting processes of plants can inspire new technologies for solar energy capture and storage, addressing future energy needs. Furthermore, understanding how plants adapt to different environmental conditions can guide the engineering of crops and microorganisms for biofuel
    """
    qa_list = parse_qa_v2(text)
    assert len(qa_list) == 6
    assert (
        qa_list[0]["question"]
        == "What is photosynthesis and why is it considered one of the most important biological processes on Earth?"
    )
    assert qa_list[-1]["answer"] == "True."


def test_qa_parser_2():
    text = """
    [Question]
    What was the primary business of the International Time Recording Company (ITR) and how did its products contribute to improving payroll processing?
    [Answer]
    The primary business of the International Time Recording Company (ITR) was accurately measuring and recording the passage of time to improve payroll processing. ITR's products, such as the Bundy Key Recorder (1888), Dial Recorder (1894), and Rochester Recorder (1894), significantly reduced inaccuracies and disputes in payroll processing. The Bundy Key Recorder printed the exact time and employee's number on a paper tape when prompted by the insertion of a specially designed key. The Dial Recorder featured employee numbers around the circumference of a large ring, allowing employees to pivot a pointer arm to their individual number and press it into a guide hole at the start and end of a shift, thereby printing the time on a sheet inside the machine. The Rochester Recorder, the world's first card-based system to record attendance, used a printed card divided into seven equal spaces representing the days of the week. When an employee inserted a card and pressed a lever, the time was printed in the appropriate \"In\" or \"Out\" slot. These innovations helped streamline and accurately record employee work hours, reducing errors and disputes in payroll processing.
    [End]
    
    [Question]
    How did the merger of the International Time Recording Company, the Computing Scale Company, and the Tabulating Machine Company in 1911 contribute to the formation of IBM?
    [Answer]
    The merger of the International Time Recording Company (ITR), the Computing Scale Company, and the Tabulating Machine Company in 1911 played a crucial role in the formation of IBM. This merger, facilitated by Charles Ranlett Flint, brought together three companies with complementary technologies. ITR, with its expertise in time recording, Computing Scale Company with its tabulating and scaling equipment, and the Tabulating Machine Company with its punch-card tabulating systems, created a diversified company with a strong foundation in data processing and information management. Following the merger, the company was renamed the Computing-Tabulating-Recording Company, and later, International Business Machines (IBM) in 1924. This strategic merger laid the groundwork for IBM's future growth and diversification into various sectors of the information technology industry.
    [End]
    
    [Question]
    What were the main challenges faced by the Computing-Tabulating-Recording Company (CTR) shortly after its formation, and how did these challenges contribute to the hiring of Thomas J. Watson as general manager?
    [Answer]
    The Computing-Tabulating-Recording Company (CTR), formed in 1911 through the merger of the International Time Recording Company, the Computing Scale Company, and the Tabulating Machine Company, faced several challenges shortly after its formation. One of the primary challenges was the significant debt accumulated from the merger, which required strategic financial management. Additionally, CTR faced resistance from managers of the merged companies who were resentful of the merger and resisted full integration. These internal conflicts hindered the company's growth and cohesion.
    
    Recognizing the need for a uniting force and a charismatic leader to overcome these challenges, Charles Ranlett Flint, the company's president, hired Thomas J. Watson as general manager in 1914. Watson's sales experience, enthusiasm, and ability to motivate employees proved invaluable in refocusing the company towards success. He implemented sales strategies that emphasized the tabulating machine, expanded operations internationally, and instilled a strong corporate culture centered around teamwork, globalism, and equal pay for equal work. These efforts helped CTR overcome its initial hurdles and set the stage for its transformation into International Business Machines (IBM).
    [End]
    """
    qa_list = parse_qa_v2(text)
    assert len(qa_list) == 3
