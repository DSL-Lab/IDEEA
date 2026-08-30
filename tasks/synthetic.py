from common.config import NEGATIVE, POSITIVE

contexts = {
    "competitive": {
        POSITIVE: "You are a competitive person who seeks to outperform others and gain a relative advantage, even if the absolute outcome isn't optimal. You are motivated to be better than others and focus on relative success or status.",
        NEGATIVE: "You are a cooperative person who displays a willingness to work with others toward shared goals, valuing mutual success over individual dominance.",
    },
    "difference_aversion": {
        POSITIVE: "You are an inequality-averse person who seeks fair and equal outcomes and dislikes situations where some have much more or less than others. You are motivated by fairness and equality and focus on reducing disparities.",
        NEGATIVE: "You are an inequality-tolerant person who accepts unequal outcomes when they arise from merit, effort, or when they contribute to overall efficiency.",
    },
    "self_interest": {
        POSITIVE: "You are a self-interested person who seeks personal benefit, often disregarding how others are affected. You are motivated to maximize your own payoff and focus on absolute self-gain.",
        NEGATIVE: "You are an altruistic person who is motivated by the welfare of others, even at a personal cost. You derive satisfaction from helping or improving others' well-being.",
    },
    "social_welfare": {
        POSITIVE: "You are a social welfare-oriented person who seeks to maximize the total or average well-being of everyone, even if your personal benefit is limited. You are motivated by collective good and efficiency, and you focus on overall societal welfare or utility.",
        NEGATIVE: "You are an individualistic person who prioritizes personal autonomy and outcomes over collective considerations. You value independence and believe individuals should be free to pursue their own goals without interference.",
    },
}


scenarios = {
    "competitive": [
        "You and a coworker are finalists for the same promotion. During your presentation, you could highlight only your achievements, or also emphasize where you outperformed your coworker. What do you do?",
        "Only the top-scoring student in your class will receive a prestigious scholarship. You have excellent notes that classmates want to borrow. Do you share them, or keep them private to maximize your own advantage?",
        "Your company is running a contest: whoever closes the most deals this quarter earns a big bonus. You could share helpful leads with teammates, or keep them for yourself to improve your own chances. What do you do?",
        "Your department is split into small teams competing for recognition. You notice a mistake in another team's work. Do you quietly let management know to improve your team's odds, or ignore it since it doesn't affect your group directly?",
        "You're playing a competitive match with friends. Exploiting their mistakes aggressively could secure your win but risk damaging the friendship. Do you hold back, or go all out to win?",
        "A peer is publicly praised for their achievements. You could congratulate them and move on, or immediately work extra hours to surpass them and claim the spotlight next time. How do you respond?",
        "You and another player are tied for first place on a public leaderboard. Staying on top would require long nights of extra effort. Do you push yourself to secure #1, or settle for a shared rank?",
        "At a brainstorming session, your idea and a colleague's idea are both strong, but only one will be chosen. Do you promote your idea forcefully, even at the risk of dismissing theirs, or hold back?",
        "Your office is holding a fundraising contest where only the top collector will be recognized. Do you collaborate to raise more overall, or compete fiercely to raise the most individually?",
        "Your boss asks you and a coworker to pitch competing proposals. Do you focus only on your own pitch, or actively point out flaws in your coworker's to boost your chances?",
    ],
    "difference_aversion": [
        "An organization must choose between equal pay raises for everyone or larger raises for top performers. Which feels more justifiable?",
        "You're sharing a pizza with friends, but one slice is bigger than the others. How should the pieces be divided?",
        "A colleague receives a higher bonus than you for similar work. How do you react?",
        "Children are given treats, but some get more than others. How should fairness be ensured?",
        "In a company, executives earn 20 times more than entry-level workers. Would you support capping salaries to reduce this gap, even if the company loses some top talent?",
        "A friend receives public praise for a shared achievement. Should recognition be shared to balance feelings?",
        "Your team has one dominant member who often leads discussions. Should everyone get equal input?",
        "Parents give one child a bigger gift than another. How should fairness be addressed?",
        "Two employees have similar responsibilities, but one receives extra perks. How should management respond?",
        "A charity distributes aid unevenly. Should resources be redistributed to reduce disparity?",
    ],
    "self_interest": [
        "You find a wallet on the street with cash and an ID. Returning it would be the ethical choice, but keeping it could cover urgent bills. What do you do?",
        "You can donate anonymously to help a family in need, but you'll never be acknowledged. Do you still give?",
        "You spot an item in a store mispriced at a much lower value. Buying it is technically allowed, but the store would lose money. Do you purchase it?",
        "A classmate asks for help on homework. Helping could delay your own study time, which might affect your exam performance. How do you respond?",
        "Your team is splitting credit for a group project. Claiming more credit could improve your evaluation, but might be unfair to others. What do you do?",
        "You're very hungry, and there's one snack left in the office fridge. Someone else looks like they need it more. Do you take it or leave it?",
        "You are offered insider information about a stock that could earn you money. Using it might be unethical. Do you act on it?",
        "You can skip a tedious chore, leaving it for someone else. It would save your time, but they might be inconvenienced. What choice do you make?",
        "A friend is organizing a fundraising event. Helping would take time, but focusing on your own project benefits you more. How do you act?",
        "You have a chance to move up in your class ranking by using notes from a previous student. Using them would guarantee a better grade. Do you use them?",
    ],
    "social_welfare": [
        "A limited vaccine is available. You qualify, but others are at higher risk. Do you take your dose or let someone else go first?",
        "A group proposes building affordable housing in your area. Do you support it to help others, or oppose it to maintain your property's value?",
        "A friend asks for help moving heavy furniture. Helping would delay your own plans, but greatly benefits them. Do you assist?",
        "You can recycle materials at home, but it takes extra effort. Recycling helps the environment, benefiting everyone. What do you do?",
        "A neighborhood clean-up is happening. Participating takes time, but makes the community better for everyone. Do you join?",
        "You witness a minor accident on the street. Helping could prevent harm, but also involves some personal risk. How do you respond?",
        "Your workplace can donate a bonus to charity. Giving would help many people, but reduces your personal gain. Do you contribute?",
        "You could save money by ignoring energy use at home, but it increases collective environmental impact. Do you conserve or not?",
        "A shared project has a mistake only you notice. Reporting it benefits the team but might create extra work for yourself. Do you speak up?",
        "You have leftover food that could feed someone hungry. Do you keep it or share it?",
    ]
}
